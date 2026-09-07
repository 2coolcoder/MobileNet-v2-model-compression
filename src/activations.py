"""Activation memory accounting (Q4b -- "state how you measured the activations").

Measurement definition
----------------------
Weights are easy to size: they are a fixed set of tensors on disk.  Activations
are not stored at all -- they are transient buffers -- so "the compression
ratio of the activations" has to be defined before it can be measured.  Two
complementary definitions are reported, both computed at **batch size 1** at
inference (``model.eval()``, no autograd, so no saved-for-backward tensors):

1. **Total activation traffic** -- the sum of the element counts of every
   tensor that crosses a quantization point, i.e. the total volume of feature
   data written to and read from memory during one forward pass.  Ratio =
   ``32 / act_bits`` weighted by how many of those tensors are actually
   quantized (tensors we deliberately leave in fp32 drag the ratio down, which
   is the honest thing for them to do).

2. **Peak concurrent activation buffer** -- the largest amount of activation
   memory that must be live at any single instant.  This is the number that
   actually decides whether the network fits on a device.  It is computed with
   an explicit liveness model over MobileNet-v2's block structure: while a
   block is executing, memory must hold (a) the block's input tensor, because
   an inverted residual needs it again for the skip addition at the very end,
   and (b) the largest intermediate tensor produced inside the block.  So

       peak = max over blocks of [ size(block input) + max size(intermediate) ]

Both are reported; the peak buffer is the headline number.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .compression.modules import ActQuant, act_quantizers


@dataclass
class ActRecord:
    name: str
    tag: str
    block: str            # top-level ``features.<i>`` this observer belongs to
    shape: tuple
    numel: int
    n_bits: int
    quantized: bool


@dataclass
class ActivationReport:
    records: List[ActRecord]
    peak_fp32_elems: int
    peak_quant_bits: float
    peak_block: str

    @property
    def total_elems(self) -> int:
        return sum(r.numel for r in self.records)

    @property
    def total_fp32_bytes(self) -> float:
        return self.total_elems * 4.0

    @property
    def total_quant_bytes(self) -> float:
        return sum(r.numel * (r.n_bits if r.quantized else 32) / 8.0
                   for r in self.records)

    @property
    def traffic_compression_ratio(self) -> float:
        return self.total_fp32_bytes / max(self.total_quant_bytes, 1e-9)

    @property
    def peak_fp32_bytes(self) -> float:
        return self.peak_fp32_elems * 4.0

    @property
    def peak_quant_bytes(self) -> float:
        return self.peak_quant_bits / 8.0

    @property
    def peak_compression_ratio(self) -> float:
        return self.peak_fp32_bytes / max(self.peak_quant_bytes, 1e-9)

    def to_dict(self) -> dict:
        return dict(
            n_quant_points=len(self.records),
            total_activation_elems=self.total_elems,
            activation_traffic_fp32_kb=self.total_fp32_bytes / 1024,
            activation_traffic_quant_kb=self.total_quant_bytes / 1024,
            activation_traffic_cr=self.traffic_compression_ratio,
            peak_buffer_fp32_kb=self.peak_fp32_bytes / 1024,
            peak_buffer_quant_kb=self.peak_quant_bytes / 1024,
            peak_buffer_cr=self.peak_compression_ratio,
            peak_block=self.peak_block,
        )


def _block_of(name: str) -> str:
    """``features.4.conv.1.2`` -> ``features.4``; classifier stays as-is."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "features":
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


@torch.no_grad()
def profile_activations(model: nn.Module, device, act_bits: int,
                        input_shape=(3, 32, 32)) -> ActivationReport:
    """Run one batch-1 forward pass and account for every quantized tensor."""
    model.eval()
    order: List[str] = []

    handles = []
    for name, mod in act_quantizers(model):
        def hook(m, inp, out, _name=name):
            order.append(_name)
        handles.append(mod.register_forward_hook(hook))
    model(torch.zeros(1, *input_shape, device=device))
    for h in handles:
        h.remove()

    lookup = dict(act_quantizers(model))
    records = [ActRecord(name=n, tag=lookup[n].tag, block=_block_of(n),
                         shape=lookup[n].shape, numel=lookup[n].numel_per_sample,
                         n_bits=lookup[n].n_bits, quantized=lookup[n].enabled)
               for n in order if lookup[n].shape is not None]

    # ---- liveness model over the top-level blocks ----------------------
    peak_elems, peak_bits, peak_block = 0, 0.0, ""
    input_elems, input_bits = 3 * 32 * 32, 3 * 32 * 32 * 32.0   # the image, fp32
    seen: Dict[str, List[ActRecord]] = {}
    for r in records:
        seen.setdefault(r.block, []).append(r)

    for block, recs in seen.items():
        largest = max(recs, key=lambda r: r.numel)
        elems = input_elems + largest.numel
        bits = input_bits + largest.numel * (largest.n_bits if largest.quantized else 32)
        if elems > peak_elems:
            peak_elems, peak_block = elems, block
        if bits > peak_bits:
            peak_bits = bits
        last = recs[-1]                       # what this block hands to the next
        input_elems = last.numel
        input_bits = last.numel * (last.n_bits if last.quantized else 32)

    return ActivationReport(records, peak_elems, peak_bits, peak_block)


def format_activation_table(rep: ActivationReport) -> str:
    d = rep.to_dict()
    return "\n".join([
        "=" * 78,
        "  ACTIVATION ACCOUNTING (batch size 1, inference)",
        "=" * 78,
        f"  quantization points              : {d['n_quant_points']}",
        f"  total activation elements        : {d['total_activation_elems']:,}",
        "-" * 78,
        f"  traffic  fp32                    : {d['activation_traffic_fp32_kb']:9.1f} KB",
        f"  traffic  quantized               : {d['activation_traffic_quant_kb']:9.1f} KB",
        f"  ACTIVATION traffic CR            : {d['activation_traffic_cr']:9.2f} x",
        "-" * 78,
        f"  peak buffer fp32   ({d['peak_block']:<12s}) : {d['peak_buffer_fp32_kb']:9.1f} KB",
        f"  peak buffer quantized            : {d['peak_buffer_quant_kb']:9.1f} KB",
        f"  ACTIVATION peak-buffer CR        : {d['peak_buffer_cr']:9.2f} x",
        "=" * 78,
    ])
