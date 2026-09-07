"""Q4/Q5 -- verification, sensitivity analysis, final operating point, figures.

Sub-commands
------------
    python analyze.py --self_test        # correctness assertions on the pipeline
    python analyze.py --sensitivity      # per-layer pruning sensitivity -> JSON
    python analyze.py --final ...        # compress + QAT the chosen config
    python analyze.py --figures          # all report figures from saved results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.activations import format_activation_table, profile_activations
from src.compression.bnfold import check_fold_equivalence, count_bn_parameters, fold_bn
from src.compression.huffman import decode_layer, encode_layer
from src.compression.modules import quant_layers
from src.compression.pipeline import calibrate, compress, convert
from src.compression.prune import (allocate_from_sensitivity, prunable_layers,
                                   sensitivity_analysis)
from src.compression.size import format_size_table, measure_model_size
from src.config import (BASELINE_CKPT, CKPT_DIR, CompressionConfig, FIG_DIR,
                        NUM_CLASSES, RESULTS_DIR, WANDB_PROJECT)
from src.data import get_calibration_loader, get_cifar10, get_heldout_loader
from src.engine import evaluate
from src.models.mobilenetv2 import mobilenet_v2
from src.utils import dump_json, seed_all, set_fp32_precision


def load_baseline(ckpt, device):
    model = mobilenet_v2(num_classes=NUM_CLASSES)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload))
    return model.to(device, memory_format=torch.channels_last).eval()


# ==========================================================================
# 1. self-test
# ==========================================================================
def run_self_test(args, device):
    """Assertions that every stage of the pipeline is doing what it claims."""
    print("=" * 70 + "\n  SELF-TEST\n" + "=" * 70)
    seed_all(args.seed)
    model = load_baseline(args.ckpt, device)
    sample = torch.randn(16, 3, 32, 32, device=device).to(memory_format=torch.channels_last)

    # -- 1. BN folding is exact -----------------------------------------
    folded = fold_bn(model)
    delta = check_fold_equivalence(model, folded, sample)
    with torch.no_grad():
        mag = model(sample).abs().max().item()
        agree = (model(sample).argmax(1) == folded(sample).argmax(1)).float().mean().item()
    rel = delta / max(mag, 1e-9)
    print(f"[1] BN fold max |logit delta|            : {delta:.3e} "
          f"(rel {rel:.2e}, predictions agree {100 * agree:.1f}%)   "
          f"({'PASS' if rel < 1e-4 and agree == 1.0 else 'FAIL'})")
    assert rel < 1e-4 and agree == 1.0, "BN folding changed the function"
    print(f"    BatchNorm fp32 values removed        : {count_bn_parameters(model):,}")

    # -- 2. Huffman round-trips on real layers --------------------------
    cfg = CompressionConfig(weight_bits=4, act_bits=8, group_size=64,
                            prune_ratio=0.5, scale_bits=8)
    qmodel = convert(model, cfg).to(device)
    from src.compression.prune import global_magnitude_prune
    global_magnitude_prune(qmodel, cfg.prune_ratio)
    for _, layer in quant_layers(qmodel):
        layer.calibrate_weight()

    checked = 0
    for name, layer in quant_layers(qmodel):
        codes = layer.integer_codes().detach().cpu().numpy().ravel().astype(np.int64)
        enc = encode_layer(codes, cfg.weight_bits, use_huffman=True, keep_blobs=True)
        if enc.scheme != "raw":
            assert np.array_equal(decode_layer(enc), codes), f"round-trip failed: {name}"
            checked += 1
    print(f"[2] Huffman encode/decode round-trips    : {checked} layers   (PASS)")

    # -- 3. size accounting is self-consistent ---------------------------
    ref_params = sum(p.numel() for p in model.parameters())
    report = measure_model_size(qmodel, cfg.weight_bits, cfg.act_bits,
                                group_size=cfg.group_size, scale_bits=cfg.scale_bits,
                                reference_params=ref_params)
    per_layer = sum(l.total_bytes for l in report.layers)
    total = report.compressed_total_bytes - report.act_param_bytes
    print(f"[3] per-layer bytes == total             : {per_layer:.1f} vs {total:.1f}   "
          f"({'PASS' if abs(per_layer - total) < 1e-6 else 'FAIL'})")
    assert abs(per_layer - total) < 1e-6

    ref_bytes = ref_params * 4
    print(f"[4] FP32 reference == params x 4         : {report.fp32_param_bytes} "
          f"vs {ref_bytes}   ({'PASS' if report.fp32_param_bytes == ref_bytes else 'FAIL'})")
    assert report.fp32_param_bytes == ref_bytes

    # -- 5. quantized codes really fit in n_bits -------------------------
    lo, hi = -(2 ** (cfg.weight_bits - 1) - 1), 2 ** (cfg.weight_bits - 1) - 1
    worst = 0
    for _, layer in quant_layers(qmodel):
        c = layer.integer_codes()
        worst = max(worst, int(c.abs().max().item()))
    print(f"[5] weight codes within [{lo}, {hi}]          : max |code| = {worst}   "
          f"({'PASS' if worst <= hi else 'FAIL'})")
    assert worst <= hi

    # -- 6. pruned weights are exactly zero ------------------------------
    nonzero_masked = sum(int(((layer.mask == 0) & (layer.integer_codes() != 0)).sum())
                         for _, layer in quant_layers(qmodel))
    print(f"[6] masked weights quantize to exactly 0 : {nonzero_masked} violations   "
          f"({'PASS' if nonzero_masked == 0 else 'FAIL'})")
    assert nonzero_masked == 0

    # -- 7. activation observers cover the graph -------------------------
    calib = get_calibration_loader(batch_size=128, num_batches=2, num_workers=2,
                                   seed=args.seed)
    calibrate(qmodel, calib, device, cfg, max_batches=2)
    act = profile_activations(qmodel, device, cfg.act_bits)
    print(f"[7] activation quantization points       : {len(act.records)} tensors, "
          f"{act.total_elems:,} elements   (PASS)")
    assert len(act.records) > 40

    print("=" * 70 + "\n  ALL SELF-TESTS PASSED\n" + "=" * 70)


# ==========================================================================
# 2. pruning sensitivity
# ==========================================================================
def run_sensitivity(args, device):
    """Per-layer sensitivity curves -> per-layer sparsity allocation."""
    seed_all(args.seed)
    model = load_baseline(args.ckpt, device)
    heldout = get_heldout_loader(n=args.heldout, num_workers=4, seed=args.seed)

    cfg = CompressionConfig(weight_bits=args.weight_bits, act_bits=args.act_bits,
                            group_size=args.group_size, scale_bits=args.scale_bits,
                            prune_ratio=0.0)
    qmodel = convert(model, cfg).to(device, memory_format=torch.channels_last)
    for _, layer in quant_layers(qmodel):
        layer.weight_quant_enabled = False        # isolate the effect of pruning

    def eval_fn(m):
        return evaluate(m, heldout, device)[1]

    print(f"Sensitivity analysis on {args.heldout} held-out training images "
          f"({len(prunable_layers(qmodel))} prunable layers)", flush=True)
    sens = sensitivity_analysis(qmodel, eval_fn)
    alloc = allocate_from_sensitivity(sens, qmodel, args.target_sparsity,
                                      tolerance=args.tolerance)

    out = {"baseline_heldout_acc": sens["__baseline__"][0.0],
           "target_sparsity": args.target_sparsity, "tolerance": args.tolerance,
           "sensitivity": {k: {str(a): b for a, b in v.items()} for k, v in sens.items()},
           "per_layer_sparsity": alloc}
    path = RESULTS_DIR / "sensitivity.json"
    dump_json(path, out)
    mean = np.mean(list(alloc.values()))
    print(f"\nAllocated sparsities: mean {mean:.3f}, "
          f"min {min(alloc.values()):.3f}, max {max(alloc.values()):.3f}")
    print(f"wrote {path}")


# ==========================================================================
# 3. final operating point
# ==========================================================================
def run_final(args, device):
    """Compress at the chosen configuration, QAT fine-tune, report everything."""
    seed_all(args.seed)
    model = load_baseline(args.ckpt, device)
    train_loader, test_loader = get_cifar10(batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            seed=args.seed, download=False)
    calib_loader = get_calibration_loader(batch_size=args.batch_size, num_batches=32,
                                          num_workers=4, seed=args.seed)
    reference_params = sum(p.numel() for p in model.parameters())
    _, baseline_acc = evaluate(model, test_loader, device)
    print(f"FP32 baseline: {baseline_acc:.2f}%", flush=True)

    per_layer = None
    if args.sparsity_map and Path(args.sparsity_map).exists():
        with open(args.sparsity_map) as fh:
            per_layer = json.load(fh)["per_layer_sparsity"]
        print(f"using per-layer sparsity map from {args.sparsity_map}")

    cfg = CompressionConfig(weight_bits=args.weight_bits, act_bits=args.act_bits,
                            group_size=args.group_size, prune_ratio=args.prune_ratio,
                            scale_bits=args.scale_bits, qat_epochs=args.qat_epochs,
                            qat_lr=args.qat_lr, calib_batches=32, seed=args.seed)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=WANDB_PROJECT, name=args.run_name, job_type="final",
                         config=cfg.as_dict())

    qmodel, size_report, info = compress(
        model, cfg, calib_loader, device, per_layer_sparsity=per_layer,
        reference_params=reference_params, train_loader=train_loader,
        test_loader=test_loader)
    _, acc = evaluate(qmodel, test_loader, device)
    act = profile_activations(qmodel, device, cfg.act_bits)

    print(f"\nFinal quantized accuracy: {acc:.2f}%  "
          f"(baseline {baseline_acc:.2f}%, drop {baseline_acc - acc:+.2f} pts)")
    print(format_size_table(size_report, max_rows=args.layer_table))
    print(format_activation_table(act))

    result = dict(config=cfg.as_dict(), baseline_acc=baseline_acc, quantized_acc=acc,
                  acc_drop=baseline_acc - acc,
                  achieved_sparsity=info["achieved_sparsity"],
                  qat_history=info.get("qat_history", []),
                  **size_report.to_dict(), **act.to_dict())
    dump_json(RESULTS_DIR / "final_result.json", result)

    import csv
    rows = size_report.to_rows()
    with open(RESULTS_DIR / "size_breakdown.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    torch.save({"state_dict": qmodel.state_dict(), "config": cfg.as_dict(),
                "result": result}, CKPT_DIR / "mobilenetv2_compressed.pth")
    print(f"\nwrote {RESULTS_DIR / 'final_result.json'}, "
          f"{RESULTS_DIR / 'size_breakdown.csv'}, "
          f"{CKPT_DIR / 'mobilenetv2_compressed.pth'}")

    if run is not None:
        run.log(result)
        run.summary.update({k: v for k, v in result.items()
                            if isinstance(v, (int, float))})
        run.finish()


# ==========================================================================
# 4. ablation of the design choices (evidence for Q2a)
# ==========================================================================
ABLATIONS = [
    ("full method",              dict()),
    ("per-tensor scales",        dict(group_size=-1)),
    ("per-channel scales",       dict(group_size=0)),
    ("+ clipping search",        dict(mse_search=True)),
    ("channel-norm pruning",     dict(prune_channel_normalized=True)),
    ("fp16 scales (no dbl-quant)", dict(scale_bits=16)),
    ("no Huffman/RLE",           dict(huffman=False)),
    ("no BN folding",            dict(fold_bn=False)),
    ("no pruning",               dict(prune_ratio=0.0)),
]


def run_ablation(args, device):
    """Turn each design choice off in turn and report what it was worth.

    ``group_size=-1`` is the "one scale for the whole tensor" case.
    """
    seed_all(args.seed)
    model = load_baseline(args.ckpt, device)
    _, test_loader = get_cifar10(batch_size=args.batch_size,
                                 num_workers=args.num_workers, seed=args.seed,
                                 download=False)
    calib_loader = get_calibration_loader(batch_size=args.batch_size, num_batches=16,
                                          num_workers=4, seed=args.seed)
    reference_params = sum(p.numel() for p in model.parameters())
    _, baseline_acc = evaluate(model, test_loader, device)

    base = dict(weight_bits=args.weight_bits, act_bits=args.act_bits,
                group_size=args.group_size, prune_ratio=args.prune_ratio,
                scale_bits=args.scale_bits, calib_batches=16, seed=args.seed)
    print(f"\nAblation at W{args.weight_bits}/A{args.act_bits} "
          f"(FP32 baseline {baseline_acc:.2f}%)\n" + "-" * 74)
    print(f"{'variant':<28s}{'acc %':>8s}{'drop':>8s}{'MB':>9s}{'ratio':>9s}"
          f"{'bits/w':>9s}")
    rows = []
    for label, override in ABLATIONS:
        kw = {**base, **override}
        cfg = CompressionConfig(**kw)
        qmodel, rep, info = compress(model, cfg, calib_loader, device,
                                     reference_params=reference_params, verbose=False)
        _, acc = evaluate(qmodel, test_loader, device)
        d = rep.to_dict()
        rows.append(dict(variant=label, acc=round(acc, 2),
                         drop=round(baseline_acc - acc, 2),
                         mb=round(d["compressed_mb"], 4),
                         ratio=round(d["model_compression_ratio"], 3),
                         bits_per_weight=round(d["avg_bits_per_weight"], 3),
                         **{k: v for k, v in cfg.as_dict().items()}))
        print(f"{label:<28s}{acc:>8.2f}{baseline_acc - acc:>8.2f}"
              f"{d['compressed_mb']:>9.4f}{d['model_compression_ratio']:>9.2f}"
              f"{d['avg_bits_per_weight']:>9.3f}", flush=True)
        del qmodel
        torch.cuda.empty_cache()

    dump_json(RESULTS_DIR / "ablation.json",
              {"baseline_acc": baseline_acc, "rows": rows})
    print(f"\nwrote {RESULTS_DIR / 'ablation.json'}")


# ==========================================================================
# 5. figures
# ==========================================================================
def run_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    from src.plots import (plot_confusion, plot_parallel_coordinates,
                           plot_pareto, plot_training_curves, plot_size_breakdown,
                           plot_sensitivity)
    made = []
    made += plot_training_curves()
    made += plot_confusion()
    made += plot_parallel_coordinates()
    made += plot_pareto()
    made += plot_size_breakdown()
    made += plot_sensitivity()
    print("figures written:")
    for m in made:
        print("  ", m)


def parse_args():
    p = argparse.ArgumentParser(description="Analysis, verification and figures")
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--figures", action="store_true")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--ckpt", type=str, default=str(BASELINE_CKPT))
    p.add_argument("--weight_bits", type=int, default=4)
    p.add_argument("--act_bits", type=int, default=8)
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--scale_bits", type=int, default=8)
    p.add_argument("--prune_ratio", type=float, default=0.5)
    p.add_argument("--qat_epochs", type=int, default=15)
    p.add_argument("--qat_lr", type=float, default=1e-3)
    p.add_argument("--sparsity_map", type=str, default=str(RESULTS_DIR / "sensitivity.json"))
    p.add_argument("--target_sparsity", type=float, default=0.6)
    p.add_argument("--tolerance", type=float, default=1.0)
    p.add_argument("--heldout", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--layer_table", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run_name", type=str, default="final-operating-point")
    return p.parse_args()


def main():
    args = parse_args()
    set_fp32_precision(exact=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.self_test:
        run_self_test(args, device)
    if args.sensitivity:
        run_sensitivity(args, device)
    if args.final:
        run_final(args, device)
    if args.ablation:
        run_ablation(args, device)
    if args.figures:
        run_figures(args)
    if not any([args.self_test, args.sensitivity, args.final, args.figures,
                args.ablation]):
        print("nothing to do -- pass one of --self_test / --sensitivity / "
              "--ablation / --final / --figures")


if __name__ == "__main__":
    main()
