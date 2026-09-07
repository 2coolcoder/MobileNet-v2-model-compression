"""Q3 -- sweep the compression knobs and log everything to Weights & Biases.

Each run is a *post-training* compression of the same trained checkpoint, so a
run costs one calibration pass plus one test-set evaluation (~30 s) and the
whole grid finishes in about an hour.  QAT is deliberately excluded here: the
sweep's job is to map the accuracy/size trade-off surface of the method, and
the chosen operating point is then fine-tuned separately in ``analyze.py``.

    python sweep.py --wandb                 # full grid, logged to W&B
    python sweep.py --dry_run               # print the grid and exit

Every run logs the columns the parallel-coordinates chart needs:
``weight_quant_bits``, ``activation_quant_bits``, ``prune_ratio``,
``group_size``, ``compression_ratio``, ``model_size_mb`` and ``quantized_acc``.
"""
import argparse
import itertools
import json
import time

import torch

from src.activations import profile_activations
from src.compression.pipeline import compress
from src.config import BASELINE_CKPT, CompressionConfig, RESULTS_DIR, WANDB_PROJECT
from src.data import get_calibration_loader, get_cifar10
from src.engine import evaluate
from src.models.mobilenetv2 import mobilenet_v2
from src.utils import CSVLogger, dump_json, seed_all, set_fp32_precision

# The grid.  Bit-widths span the interesting range (8 -> near-lossless, 2 ->
# ternary); sparsity and grouping are the two knobs unique to our method.
DEFAULT_GRID = {
    "weight_bits": [2, 3, 4, 5, 6, 8],
    "act_bits": [4, 6, 8],
    "prune_ratio": [0.0, 0.3, 0.5],
    "group_size": [0, 64],
}

COLUMNS = ["weight_quant_bits", "activation_quant_bits", "prune_ratio", "group_size",
           "quantized_acc", "acc_drop", "compression_ratio", "weight_cr",
           "activation_traffic_cr", "activation_peak_cr", "model_size_mb",
           "avg_bits_per_weight", "achieved_sparsity", "metadata_fraction",
           "runtime_s"]


def parse_args():
    p = argparse.ArgumentParser(description="Compression sweep for the W&B chart")
    p.add_argument("--ckpt", type=str, default=str(BASELINE_CKPT))
    p.add_argument("--weight_bits", type=int, nargs="+", default=DEFAULT_GRID["weight_bits"])
    p.add_argument("--act_bits", type=int, nargs="+", default=DEFAULT_GRID["act_bits"])
    p.add_argument("--prune_ratio", type=float, nargs="+", default=DEFAULT_GRID["prune_ratio"])
    p.add_argument("--group_size", type=int, nargs="+", default=DEFAULT_GRID["group_size"])
    p.add_argument("--scale_bits", type=int, default=8)
    p.add_argument("--calib_batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sparsity_map", type=str, default=None,
                   help="JSON from analyze.py --sensitivity, for per-layer sparsity")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--group", type=str, default="ptq-grid")
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "sweep_results.csv"))
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    combos = list(itertools.product(args.weight_bits, args.act_bits,
                                    args.prune_ratio, args.group_size))
    print(f"{len(combos)} configurations to evaluate")
    if args.dry_run:
        for c in combos:
            print("  W%d A%d prune=%.2f group=%s" % c)
        return

    seed_all(args.seed)
    set_fp32_precision(exact=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_cifar10(batch_size=args.batch_size, num_workers=args.num_workers,
                                 seed=args.seed, download=False)
    calib_loader = get_calibration_loader(batch_size=args.batch_size,
                                          num_batches=args.calib_batches,
                                          num_workers=4, seed=args.seed)

    model = mobilenet_v2()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload))
    model.to(device, memory_format=torch.channels_last).eval()
    reference_params = sum(p.numel() for p in model.parameters())

    _, baseline_acc = evaluate(model, test_loader, device)
    print(f"FP32 baseline: {baseline_acc:.2f}%  ({reference_params:,} params, "
          f"{reference_params * 4 / 1024 ** 2:.2f} MB)\n", flush=True)

    per_layer = None
    if args.sparsity_map:
        with open(args.sparsity_map) as fh:
            per_layer = json.load(fh)["per_layer_sparsity"]

    logger = CSVLogger(args.out, COLUMNS)
    rows = []
    for i, (wb, ab, pr, gs) in enumerate(combos, 1):
        t0 = time.time()
        cfg = CompressionConfig(weight_bits=wb, act_bits=ab, prune_ratio=pr,
                                group_size=gs, scale_bits=args.scale_bits,
                                calib_batches=args.calib_batches, seed=args.seed)
        qmodel, size_report, info = compress(
            model, cfg, calib_loader, device, per_layer_sparsity=per_layer,
            reference_params=reference_params, verbose=False)
        _, acc = evaluate(qmodel, test_loader, device)
        act = profile_activations(qmodel, device, ab)
        sd = size_report.to_dict()

        row = dict(weight_quant_bits=wb, activation_quant_bits=ab, prune_ratio=pr,
                   group_size=gs, quantized_acc=round(acc, 3),
                   acc_drop=round(baseline_acc - acc, 3),
                   compression_ratio=round(sd["model_compression_ratio"], 4),
                   weight_cr=round(sd["weight_compression_ratio"], 4),
                   activation_traffic_cr=round(act.traffic_compression_ratio, 4),
                   activation_peak_cr=round(act.peak_compression_ratio, 4),
                   model_size_mb=round(sd["compressed_mb"], 5),
                   avg_bits_per_weight=round(sd["avg_bits_per_weight"], 4),
                   achieved_sparsity=round(info["achieved_sparsity"], 4),
                   metadata_fraction=round(sd["metadata_fraction"], 4),
                   runtime_s=round(time.time() - t0, 1))
        logger.log(row)
        rows.append(row)
        print(f"[{i:3d}/{len(combos)}] W{wb} A{ab} prune={pr:.2f} "
              f"group={gs or 'chan':>4} | acc {acc:6.2f}% | "
              f"{sd['compressed_mb']:6.3f} MB | {sd['model_compression_ratio']:6.2f}x "
              f"| {row['runtime_s']:.0f}s", flush=True)

        if args.wandb:
            import wandb
            run = wandb.init(project=WANDB_PROJECT, group=args.group,
                             name=f"W{wb}A{ab}-p{pr}-g{gs or 'chan'}",
                             job_type="sweep", config=cfg.as_dict(), reinit=True)
            run.log({**row, "baseline_acc": baseline_acc})
            run.summary.update(row)
            run.finish()

        del qmodel
        torch.cuda.empty_cache()

    dump_json(RESULTS_DIR / "sweep_results.json",
              {"baseline_acc": baseline_acc, "rows": rows})
    best = max(rows, key=lambda r: r["compression_ratio"] if r["acc_drop"] < 1.5 else -1)
    print(f"\nWrote {args.out}")
    print(f"Best config within 1.5 pts of baseline: {best}")


if __name__ == "__main__":
    main()
