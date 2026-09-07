"""Push an already-computed sweep (results/sweep_results.csv) to Weights & Biases.

Creating a W&B run costs far more wall-clock than evaluating a compression
configuration does (~85 s versus ~3 s), so ``sweep.py`` computes the whole grid
offline and this script uploads it afterwards.  One run per configuration is
what the parallel-coordinates panel needs: it plots across runs, reading each
run's config and summary.

    python sweep.py                 # compute the grid  (~6 min)
    python upload_sweep.py          # create the W&B runs
"""
import argparse
import csv
import json
import time
from pathlib import Path

import wandb

from src.config import RESULTS_DIR, WANDB_PROJECT

# Columns that describe *what was configured* rather than what came out.
CONFIG_COLS = ("weight_quant_bits", "activation_quant_bits", "prune_ratio",
               "group_size")


def fast_settings():
    """Strip everything from a run that we do not need: system metrics, the
    environment probe, console capture and code upload."""
    for kwargs in ({"x_disable_stats": True, "x_disable_meta": True},
                   {"_disable_stats": True, "_disable_meta": True},
                   {}):
        try:
            return wandb.Settings(console="off", save_code=False, **kwargs)
        except (TypeError, ValueError):
            continue
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Upload a computed sweep to W&B")
    p.add_argument("--csv", default=str(RESULTS_DIR / "sweep_results.csv"))
    p.add_argument("--group", default="ptq-grid")
    p.add_argument("--project", default=WANDB_PROJECT)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.csv) as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    summary_path = RESULTS_DIR / "sweep_results.json"
    baseline_acc = None
    if summary_path.exists():
        baseline_acc = json.loads(summary_path.read_text()).get("baseline_acc")

    settings = fast_settings()
    print(f"uploading {len(rows)} runs to {args.project}", flush=True)
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        numeric = {k: float(v) for k, v in row.items() if v not in ("", None)}
        config = {k: numeric[k] for k in CONFIG_COLS if k in numeric}
        name = (f"W{int(numeric['weight_quant_bits'])}"
                f"A{int(numeric['activation_quant_bits'])}"
                f"-p{numeric['prune_ratio']:g}"
                f"-g{int(numeric['group_size']) or 'chan'}")
        run = wandb.init(project=args.project, group=args.group, name=name,
                         job_type="sweep", config=config, reinit=True,
                         settings=settings)
        if baseline_acc is not None:
            numeric["baseline_acc"] = baseline_acc
        run.log(numeric)
        run.summary.update(numeric)
        run.finish()
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  ({(time.time() - t0) / i:.1f}s per run)",
                  flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
