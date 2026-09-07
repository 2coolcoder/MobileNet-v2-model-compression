"""Evaluate the compression pipeline at a given operating point.

The interface required by the assignment works verbatim:

    python test.py --weight_quant_bits 8 --activation_quant_bits 8

and every other knob of the method is exposed as an extra flag:

    python test.py --weight_quant_bits 4 --activation_quant_bits 8 \
                   --group_size 64 --prune_ratio 0.5 --qat_epochs 15
"""
import argparse
import json

import torch

from src.activations import format_activation_table, profile_activations
from src.compression.pipeline import compress
from src.compression.size import format_size_table
from src.config import (BASELINE_CKPT, CompressionConfig, NUM_CLASSES, RESULTS_DIR,
                        WANDB_PROJECT)
from src.data import get_calibration_loader, get_cifar10
from src.engine import evaluate
from src.models.mobilenetv2 import mobilenet_v2
from src.utils import dump_json, seed_all, set_fp32_precision


def build_parser() -> argparse.ArgumentParser:
    cfg = CompressionConfig()
    p = argparse.ArgumentParser(description="Compress and evaluate MobileNet-v2")
    # --- names mandated by the assignment ---
    p.add_argument("--weight_quant_bits", type=int, default=cfg.weight_bits,
                   help="bit-width of the quantized weights")
    p.add_argument("--activation_quant_bits", type=int, default=cfg.act_bits,
                   help="bit-width of the quantized activations")
    # --- the rest of our method ---
    p.add_argument("--group_size", type=int, default=cfg.group_size,
                   help="0 = one scale per output channel, >0 = per group of N weights")
    p.add_argument("--prune_ratio", type=float, default=cfg.prune_ratio,
                   help="target sparsity over the prunable (pointwise) layers")
    p.add_argument("--sparsity_map", type=str, default=None,
                   help="JSON with per-layer sparsities from the sensitivity analysis")
    p.add_argument("--scale_bits", type=int, default=cfg.scale_bits,
                   help="bits per quantization scale (<16 enables log-domain "
                        "double quantization of the scales)")
    p.add_argument("--bias_bits", type=int, default=cfg.bias_bits,
                   help="bits per bias value (16 = fp16)")
    p.add_argument("--no_huffman", action="store_true", help="skip entropy coding")
    p.add_argument("--no_fold_bn", action="store_true", help="keep BatchNorm separate")
    p.add_argument("--clip_search", action="store_true",
                   help="search the weight clipping range instead of using plain "
                        "min/max (measured to cost accuracy on MobileNet-v2)")
    p.add_argument("--calib_batches", type=int, default=cfg.calib_batches)
    p.add_argument("--qat_epochs", type=int, default=cfg.qat_epochs)
    p.add_argument("--qat_lr", type=float, default=cfg.qat_lr)
    # --- plumbing ---
    p.add_argument("--ckpt", type=str, default=str(BASELINE_CKPT))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--skip_baseline", action="store_true",
                   help="do not evaluate the fp32 model first (saves ~10s)")
    p.add_argument("--save_json", type=str, default=None)
    p.add_argument("--save_ckpt", type=str, default=None)
    p.add_argument("--layer_table", type=int, default=0,
                   help="print the first N rows of the per-layer size breakdown")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run_name", type=str, default=None)
    return p


def config_from_args(args) -> CompressionConfig:
    return CompressionConfig(
        weight_bits=args.weight_quant_bits, act_bits=args.activation_quant_bits,
        group_size=args.group_size, prune_ratio=args.prune_ratio,
        huffman=not args.no_huffman, fold_bn=not args.no_fold_bn,
        mse_search=args.clip_search, calib_batches=args.calib_batches,
        scale_bits=args.scale_bits, bias_bits=args.bias_bits,
        qat_epochs=args.qat_epochs, qat_lr=args.qat_lr, seed=args.seed)


def main():
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    seed_all(args.seed)
    set_fp32_precision(exact=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10(batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            seed=args.seed, download=False)
    calib_loader = get_calibration_loader(batch_size=args.batch_size,
                                          num_batches=cfg.calib_batches,
                                          num_workers=min(4, args.num_workers),
                                          seed=args.seed)

    model = mobilenet_v2(num_classes=NUM_CLASSES)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload))
    model.to(device, memory_format=torch.channels_last).eval()
    reference_params = sum(p.numel() for p in model.parameters())

    baseline_acc = float("nan")
    if not args.skip_baseline:
        _, baseline_acc = evaluate(model, test_loader, device)
        print(f"FP32 baseline Test Acc = {baseline_acc:.2f}%", flush=True)

    per_layer = None
    if args.sparsity_map:
        with open(args.sparsity_map) as fh:
            per_layer = json.load(fh)["per_layer_sparsity"]

    print(f"\nCompressing: W{cfg.weight_bits} / A{cfg.act_bits} | "
          f"group_size={cfg.group_size or 'per-channel'} | prune={cfg.prune_ratio} | "
          f"huffman={cfg.huffman} | fold_bn={cfg.fold_bn} | qat={cfg.qat_epochs}",
          flush=True)
    qmodel, size_report, info = compress(
        model, cfg, calib_loader, device, per_layer_sparsity=per_layer,
        reference_params=reference_params,
        train_loader=train_loader, test_loader=test_loader)

    _, quant_acc = evaluate(qmodel, test_loader, device)
    act_report = profile_activations(qmodel, device, cfg.act_bits)

    print(f"\nQuantized Test Acc = {quant_acc:.2f}%"
          + ("" if args.skip_baseline else f"   (drop {baseline_acc - quant_acc:+.2f} pts)"))
    print(format_size_table(size_report, max_rows=args.layer_table))
    print(format_activation_table(act_report))

    result = dict(config=cfg.as_dict(), baseline_acc=baseline_acc,
                  quantized_acc=quant_acc, acc_drop=baseline_acc - quant_acc,
                  achieved_sparsity=info["achieved_sparsity"],
                  **size_report.to_dict(), **act_report.to_dict())
    if args.save_json:
        dump_json(args.save_json, result)
        print(f"\nwrote {args.save_json}")
    if args.save_ckpt:
        torch.save({"state_dict": qmodel.state_dict(), "config": cfg.as_dict(),
                    "result": result}, args.save_ckpt)
        print(f"wrote {args.save_ckpt}")

    if args.wandb:
        import wandb
        name = args.run_name or f"W{cfg.weight_bits}A{cfg.act_bits}-p{cfg.prune_ratio}"
        run = wandb.init(project=WANDB_PROJECT, name=name, job_type="compress",
                         config=cfg.as_dict())
        run.log(result)
        run.finish()
    return result


if __name__ == "__main__":
    main()
