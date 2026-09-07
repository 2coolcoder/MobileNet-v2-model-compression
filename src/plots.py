"""Report figures.  Every plot reads a file written by train/sweep/analyze."""
import csv
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASSES, FIG_DIR, RESULTS_DIR

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight",
})

ACCENT = "#2b6cb0"
ACCENT2 = "#c05621"


def _save(fig, name: str) -> str:
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _read_csv(path):
    with open(path) as fh:
        return [dict(r) for r in csv.DictReader(fh)]


# --------------------------------------------------------------- Q1c ------
def plot_training_curves() -> List[str]:
    path = RESULTS_DIR / "train_log.csv"
    if not path.exists():
        return []
    rows = _read_csv(path)
    ep = [int(r["epoch"]) for r in rows]
    out = []

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(ep, [float(r["train_loss"]) for r in rows], label="train", color=ACCENT)
    ax.plot(ep, [float(r["test_loss"]) for r in rows], label="test", color=ACCENT2)
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy (label-smoothed)")
    ax.set_title("MobileNet-v2 / CIFAR-10 -- loss"); ax.legend()
    out.append(_save(fig, "curves_loss.png"))

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(ep, [float(r["train_acc"]) for r in rows], label="train", color=ACCENT)
    ax.plot(ep, [float(r["test_acc"]) for r in rows], label="test", color=ACCENT2)
    best = max(float(r["test_acc"]) for r in rows)
    ax.axhline(best, ls="--", lw=0.8, color="grey")
    ax.annotate(f"best {best:.2f}%", (ep[len(ep) // 6], best + 0.6), color="grey")
    ax.set_xlabel("epoch"); ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title("MobileNet-v2 / CIFAR-10 -- accuracy"); ax.legend(loc="lower right")
    out.append(_save(fig, "curves_acc.png"))
    return out


def plot_confusion() -> List[str]:
    path = RESULTS_DIR / "baseline_summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    cm = np.array(data["confusion_matrix"], dtype=float)
    cm_norm = cm / cm.sum(1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (% of each true class)")
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            v = cm_norm[i, j]
            if v >= 1.0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return [_save(fig, "confusion_matrix.png")]


# --------------------------------------------------------------- Q3 -------
PC_COLUMNS = [("activation_quant_bits", "act bits"), ("weight_quant_bits", "weight bits"),
              ("prune_ratio", "prune ratio"), ("compression_ratio", "compression ratio"),
              ("model_size_mb", "model size (MB)"), ("quantized_acc", "accuracy (%)")]


def plot_parallel_coordinates() -> List[str]:
    """Local twin of the W&B parallel-coordinates chart (Q3b)."""
    path = RESULTS_DIR / "sweep_results.csv"
    if not path.exists():
        return []
    rows = _read_csv(path)
    cols = [c for c, _ in PC_COLUMNS]
    data = np.array([[float(r[c]) for c in cols] for r in rows])

    lo, hi = data.min(0), data.max(0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    norm = (data - lo) / span

    acc = data[:, cols.index("quantized_acc")]
    cmap = plt.get_cmap("viridis")
    shade = (acc - acc.min()) / max(acc.max() - acc.min(), 1e-9)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    order = np.argsort(acc)                      # best runs drawn on top
    for i in order:
        ax.plot(range(len(cols)), norm[i], color=cmap(shade[i]), alpha=0.55, lw=1.0)
    for k in range(len(cols)):
        ax.axvline(k, color="black", lw=0.8, alpha=0.5)
        for frac in np.linspace(0, 1, 6):
            ax.text(k, frac, f" {lo[k] + frac * span[k]:.4g}", fontsize=6.5,
                    va="center", ha="left", color="#444")
    ax.set_xticks(range(len(cols)), [lbl for _, lbl in PC_COLUMNS])
    ax.set_yticks([])
    ax.set_ylim(-0.06, 1.10)
    ax.grid(False)
    ax.set_title(f"Compression sweep -- {len(rows)} configurations "
                 "(colour = post-compression accuracy)")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                 norm=plt.Normalize(acc.min(), acc.max())),
                 ax=ax, fraction=0.02, label="top-1 (%)")
    return [_save(fig, "parallel_coords.png")]


def plot_pareto() -> List[str]:
    path = RESULTS_DIR / "sweep_results.csv"
    if not path.exists():
        return []
    rows = _read_csv(path)
    cr = np.array([float(r["compression_ratio"]) for r in rows])
    acc = np.array([float(r["quantized_acc"]) for r in rows])
    wb = np.array([int(r["weight_quant_bits"]) for r in rows])

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    sc = ax.scatter(cr, acc, c=wb, cmap="plasma", s=26, alpha=0.85,
                    edgecolor="white", linewidth=0.4)
    # Pareto front: maximise both
    order = np.argsort(-cr)
    front_x, front_y, best = [], [], -np.inf
    for i in order:
        if acc[i] > best:
            best = acc[i]
            front_x.append(cr[i]); front_y.append(acc[i])
    ax.plot(front_x, front_y, color="black", lw=1.0, ls="--", label="Pareto front")
    ax.set_xlabel("model compression ratio (x)"); ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title("Accuracy vs. compression (PTQ sweep)")
    ax.legend(loc="lower left")
    fig.colorbar(sc, ax=ax, label="weight bits")
    return [_save(fig, "pareto.png")]


# --------------------------------------------------------------- Q2c ------
def plot_size_breakdown() -> List[str]:
    fpath = RESULTS_DIR / "final_result.json"
    if not fpath.exists():
        return []
    d = json.loads(fpath.read_text())
    parts = [("weight payload", d["payload_kb"]), ("scales", d["scale_kb"]),
             ("Huffman tables", d["huffman_table_kb"]), ("biases", d["bias_kb"]),
             ("headers", d["header_kb"]), ("activation params", d["act_param_kb"])]
    parts = [(n, v) for n, v in parts if v > 0]
    out = []

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    names = [n for n, _ in parts]; vals = [v for _, v in parts]
    colors = [ACCENT] + [ACCENT2] * (len(parts) - 1)
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:.1f} KB ({100 * v / sum(vals):.1f}%)", va="center", fontsize=7.5)
    ax.set_xlabel("KB"); ax.set_xlim(0, max(vals) * 1.45)
    ax.set_title(f"Compressed model = {d['compressed_mb']:.3f} MB "
                 f"({d['model_compression_ratio']:.2f}x)")
    out.append(_save(fig, "size_breakdown.png"))

    cpath = RESULTS_DIR / "size_breakdown.csv"
    if cpath.exists():
        rows = _read_csv(cpath)
        kinds = sorted({r["kind"] for r in rows})
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for kind, color in zip(kinds, plt.get_cmap("tab10").colors):
            sel = [r for r in rows if r["kind"] == kind]
            ax.scatter([int(r["n_weights"]) for r in sel],
                       [float(r["bits_per_weight"]) for r in sel],
                       s=22, label=kind, color=color, alpha=0.85)
        ax.set_xscale("log"); ax.set_xlabel("weights in layer")
        ax.set_ylabel("stored bits / weight (incl. metadata)")
        ax.set_title("Per-layer storage cost after compression")
        ax.legend(fontsize=7.5)
        out.append(_save(fig, "bits_per_layer.png"))
    return out


def plot_sensitivity() -> List[str]:
    path = RESULTS_DIR / "sensitivity.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    base = d["baseline_heldout_acc"]
    sens = {k: v for k, v in d["sensitivity"].items() if k != "__baseline__"}
    levels = sorted(float(x) for x in next(iter(sens.values())).keys())

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cmap = plt.get_cmap("viridis")
    for i, (name, curve) in enumerate(sens.items()):
        ax.plot(levels, [base - curve[str(l)] for l in levels],
                color=cmap(i / max(len(sens) - 1, 1)), lw=0.9, alpha=0.75)
    ax.axhline(d["tolerance"], color="red", ls="--", lw=1.0,
               label=f"tolerance = {d['tolerance']} pts")
    ax.set_xlabel("per-layer sparsity"); ax.set_ylabel("held-out accuracy drop (pts)")
    ax.set_title(f"Pruning sensitivity of the {len(sens)} prunable layers")
    ax.set_yscale("symlog", linthresh=1.0); ax.legend()
    return [_save(fig, "sensitivity.png")]
