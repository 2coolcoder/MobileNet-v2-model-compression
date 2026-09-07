"""Render the result JSON/CSV files as Markdown tables for REPORT.md.

Keeping this in code means every number in the report is generated from the
artefacts in ``results/`` and cannot drift from them.

    python -m src.tables               # print every available table
    python -m src.tables sweep         # just one
"""
import csv
import json
import sys
from pathlib import Path

from .config import RESULTS_DIR


def _load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    with open(p) as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _md(headers, rows, align=None):
    align = align or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def baseline_table() -> str:
    d = _load("baseline_summary.json")
    if not d:
        return ""
    cls = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
    rows = [(c, f"{a:.1f}") for c, a in zip(cls, d["per_class_acc"])]
    return ("**Per-class top-1 accuracy (%)**\n\n"
            + _md(["class", "accuracy"], rows, ["---", "---:"]))


def confusions_table(top: int = 6) -> str:
    d = _load("baseline_summary.json")
    if not d:
        return ""
    cm = d["confusion_matrix"]
    cls = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
    pairs = []
    for i in range(10):
        total = sum(cm[i])
        for j in range(10):
            if i != j:
                pairs.append((cm[i][j] / total * 100, cls[i], cls[j]))
    pairs.sort(reverse=True)
    rows = [(t, p, f"{v:.1f}") for v, t, p in pairs[:top]]
    return ("**Most frequent confusions**\n\n"
            + _md(["true", "predicted as", "% of class"], rows, ["---", "---", "---:"]))


def ablation_table() -> str:
    d = _load("ablation.json")
    if not d:
        return ""
    rows = [(r["variant"], f"{r['acc']:.2f}", f"{r['drop']:+.2f}",
             f"{r['mb']:.3f}", f"{r['ratio']:.2f}x", f"{r['bits_per_weight']:.3f}")
            for r in d["rows"]]
    return (f"FP32 baseline: **{d['baseline_acc']:.2f}%**\n\n"
            + _md(["variant", "top-1 %", "drop", "size MB", "model CR", "bits/weight"],
                  rows, ["---", "---:", "---:", "---:", "---:", "---:"]))


def sweep_table(top: int = 15, sort_key: str = "compression_ratio") -> str:
    rows = _load("sweep_results.csv")
    if not rows:
        return ""
    rows = sorted(rows, key=lambda r: -float(r[sort_key]))[:top]
    out = [(r["weight_quant_bits"], r["activation_quant_bits"], r["prune_ratio"],
            r["group_size"] if r["group_size"] != "0" else "per-chan",
            f"{float(r['quantized_acc']):.2f}", f"{float(r['model_size_mb']):.3f}",
            f"{float(r['compression_ratio']):.2f}x",
            f"{float(r['activation_traffic_cr']):.2f}x") for r in rows]
    return _md(["W bits", "A bits", "prune", "group", "top-1 %", "MB",
                "model CR", "act CR"], out,
               ["---:"] * 4 + ["---:"] * 4)


def pareto_table(max_drop: float = 2.0) -> str:
    rows = _load("sweep_results.csv")
    if not rows:
        return ""
    keep = [r for r in rows if float(r["acc_drop"]) <= max_drop]
    keep.sort(key=lambda r: -float(r["compression_ratio"]))
    out = [(r["weight_quant_bits"], r["activation_quant_bits"], r["prune_ratio"],
            r["group_size"] if r["group_size"] != "0" else "per-chan",
            f"{float(r['quantized_acc']):.2f}", f"{float(r['acc_drop']):+.2f}",
            f"{float(r['model_size_mb']):.3f}",
            f"{float(r['compression_ratio']):.2f}x") for r in keep[:12]]
    return _md(["W bits", "A bits", "prune", "group", "top-1 %", "drop", "MB",
                "model CR"], out, ["---:"] * 8)


def final_size_table() -> str:
    d = _load("final_result.json")
    if not d:
        return ""
    total = d["compressed_mb"] * 1024
    parts = [("quantized weight payload", d["payload_kb"]),
             ("quantization scales", d["scale_kb"]),
             ("Huffman code books", d["huffman_table_kb"]),
             ("biases (fp16)", d["bias_kb"]),
             ("per-layer headers", d["header_kb"]),
             ("activation scales / zero-points", d["act_param_kb"])]
    rows = [(n, f"{v:.1f}", f"{100 * v / total:.2f}") for n, v in parts if v > 0]
    rows.append(("**total**", f"**{total:.1f}**", "**100.00**"))
    return _md(["component", "KB", "% of compressed model"], rows,
               ["---", "---:", "---:"])


def final_summary_table() -> str:
    d = _load("final_result.json")
    if not d:
        return ""
    c = d["config"]
    rows = [
        ("FP32 baseline top-1", f"{d['baseline_acc']:.2f} %"),
        ("compressed top-1", f"{d['quantized_acc']:.2f} %"),
        ("accuracy drop", f"{d['acc_drop']:+.2f} pts"),
        ("FP32 model size", f"{d['fp32_mb']:.3f} MB"),
        ("compressed model size", f"**{d['compressed_mb']:.3f} MB**"),
        ("weight compression ratio", f"**{d['weight_compression_ratio']:.2f}x**"),
        ("model compression ratio", f"**{d['model_compression_ratio']:.2f}x**"),
        ("activation CR (traffic)", f"{d['activation_traffic_cr']:.2f}x"),
        ("activation CR (peak buffer)", f"**{d['peak_buffer_cr']:.2f}x**"),
        ("mean bits / weight (incl. metadata)", f"{d['avg_bits_per_weight']:.3f}"),
        ("weight sparsity", f"{100 * d['overall_sparsity']:.2f} %"),
        ("metadata share of model", f"{100 * d['metadata_fraction']:.2f} %"),
        ("configuration", f"W{c['weight_bits']} / A{c['act_bits']}, "
                          f"group={c['group_size']}, scale_bits={c['scale_bits']}, "
                          f"prune={c['prune_ratio']}, QAT={c['qat_epochs']} ep"),
    ]
    return _md(["quantity", "value"], rows, ["---", "---:"])


def layer_kind_table() -> str:
    rows = _load("size_breakdown.csv")
    if not rows:
        return ""
    kinds = {}
    for r in rows:
        k = kinds.setdefault(r["kind"], dict(n=0, w=0, kb=0.0, fp32=0.0, sp=0.0))
        k["n"] += 1
        k["w"] += int(r["n_weights"])
        k["kb"] += float(r["total_kb"])
        k["fp32"] += float(r["fp32_kb"])
        k["sp"] += float(r["sparsity"]) * int(r["n_weights"])
    out = []
    for kind, v in sorted(kinds.items(), key=lambda kv: -kv[1]["w"]):
        out.append((kind, v["n"], f"{v['w']:,}", f"{100 * v['sp'] / v['w']:.1f}",
                    f"{v['fp32']:.1f}", f"{v['kb']:.1f}",
                    f"{v['fp32'] / max(v['kb'], 1e-9):.2f}x",
                    f"{v['kb'] * 8 * 1024 / v['w']:.2f}"))
    return _md(["layer kind", "#layers", "#weights", "sparsity %", "fp32 KB",
                "compressed KB", "CR", "bits/weight"], out,
               ["---"] + ["---:"] * 7)


TABLES = {
    "baseline": baseline_table, "confusions": confusions_table,
    "ablation": ablation_table, "sweep": sweep_table, "pareto": pareto_table,
    "final_size": final_size_table, "final_summary": final_summary_table,
    "layer_kind": layer_kind_table,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(TABLES)
    for n in names:
        body = TABLES[n]()
        if body:
            print(f"\n### {n}\n\n{body}\n")
        else:
            print(f"\n### {n}\n\n(no data yet)\n")
