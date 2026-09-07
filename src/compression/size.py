"""Byte-exact model size accounting, including every storage overhead (Q2c).

The rule followed everywhere in this file: *if a decoder would need it, it is
charged*.  That covers the quantized weight payload, the per-channel/per-group
scales, the Huffman code books, the biases, the activation quantization
parameters and a small per-layer header.  Nothing is hand-waved away, so the
compression ratios reported in the reportare reproducible from these numbers
alone.

Baseline convention
-------------------
The FP32 reference is the *unfolded, unpruned* model's parameters at 4 bytes
each (2,236,682 params -> 8.53 MB).  BatchNorm running statistics
(34,112 fp32 values, 0.13 MB) are listed separately and **excluded** from the
numerator, which makes every ratio quoted here slightly conservative rather
than flattering.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .huffman import EncodedLayer, encode_layer
from .modules import ActQuant, QuantConv2d, QuantLinear, act_quantizers, quant_layers

# Per-layer header stored alongside the payload: bit-width, group size, scheme
# id and symbol count.  Four bytes is generous but keeps the accounting honest.
LAYER_HEADER_BYTES = 4


@dataclass
class LayerSize:
    name: str
    kind: str
    n_weights: int
    n_bits: int
    sparsity: float
    scheme: str
    payload_bits: int
    scale_bytes: int
    table_bytes: int
    bias_bytes: int
    header_bytes: int = LAYER_HEADER_BYTES

    @property
    def fp32_weight_bytes(self) -> int:
        return self.n_weights * 4

    @property
    def weight_bytes(self) -> float:
        """Everything needed to reconstruct this layer's weight tensor."""
        return (self.payload_bits / 8.0 + self.scale_bytes + self.table_bytes
                + self.header_bytes)

    @property
    def total_bytes(self) -> float:
        return self.weight_bytes + self.bias_bytes

    @property
    def bits_per_weight(self) -> float:
        return self.weight_bytes * 8 / max(self.n_weights, 1)


@dataclass
class SizeReport:
    layers: List[LayerSize] = field(default_factory=list)
    act_param_bytes: int = 0
    bn_bytes: int = 0
    fp32_param_bytes: int = 0
    fp32_buffer_bytes: int = 0
    n_params: int = 0

    # ---- aggregate weight numbers -------------------------------------
    @property
    def fp32_weight_bytes(self) -> int:
        return sum(l.fp32_weight_bytes for l in self.layers)

    @property
    def compressed_weight_bytes(self) -> float:
        return sum(l.weight_bytes for l in self.layers)

    @property
    def bias_bytes(self) -> float:
        return sum(l.bias_bytes for l in self.layers)

    @property
    def scale_bytes(self) -> int:
        return sum(l.scale_bytes for l in self.layers)

    @property
    def table_bytes(self) -> int:
        return sum(l.table_bytes for l in self.layers)

    @property
    def header_bytes(self) -> int:
        return sum(l.header_bytes for l in self.layers)

    @property
    def payload_bytes(self) -> float:
        return sum(l.payload_bits for l in self.layers) / 8.0

    @property
    def metadata_bytes(self) -> float:
        """Everything that is *not* the quantized weight payload."""
        return (self.scale_bytes + self.table_bytes + self.header_bytes
                + self.bias_bytes + self.act_param_bytes + self.bn_bytes)

    @property
    def compressed_total_bytes(self) -> float:
        return (self.compressed_weight_bytes + self.bias_bytes
                + self.act_param_bytes + self.bn_bytes)

    # ---- ratios --------------------------------------------------------
    @property
    def weight_compression_ratio(self) -> float:
        return self.fp32_weight_bytes / max(self.compressed_weight_bytes, 1e-9)

    @property
    def model_compression_ratio(self) -> float:
        return self.fp32_param_bytes / max(self.compressed_total_bytes, 1e-9)

    @property
    def compressed_mb(self) -> float:
        return self.compressed_total_bytes / 1024 ** 2

    @property
    def fp32_mb(self) -> float:
        return self.fp32_param_bytes / 1024 ** 2

    @property
    def metadata_fraction(self) -> float:
        return self.metadata_bytes / max(self.compressed_total_bytes, 1e-9)

    @property
    def overall_sparsity(self) -> float:
        tot = sum(l.n_weights for l in self.layers)
        return sum(l.sparsity * l.n_weights for l in self.layers) / max(tot, 1)

    def to_dict(self) -> dict:
        return dict(
            fp32_mb=self.fp32_mb,
            compressed_mb=self.compressed_mb,
            weight_compression_ratio=self.weight_compression_ratio,
            model_compression_ratio=self.model_compression_ratio,
            payload_kb=self.payload_bytes / 1024,
            scale_kb=self.scale_bytes / 1024,
            huffman_table_kb=self.table_bytes / 1024,
            bias_kb=self.bias_bytes / 1024,
            header_kb=self.header_bytes / 1024,
            act_param_kb=self.act_param_bytes / 1024,
            bn_kb=self.bn_bytes / 1024,
            metadata_kb=self.metadata_bytes / 1024,
            metadata_fraction=self.metadata_fraction,
            overall_sparsity=self.overall_sparsity,
            avg_bits_per_weight=self.compressed_weight_bytes * 8 /
                                max(sum(l.n_weights for l in self.layers), 1),
            n_params=self.n_params,
        )

    def to_rows(self) -> List[dict]:
        return [dict(layer=l.name, kind=l.kind, n_weights=l.n_weights,
                     bits=l.n_bits, sparsity=round(l.sparsity, 4), scheme=l.scheme,
                     payload_kb=round(l.payload_bits / 8 / 1024, 3),
                     scale_kb=round(l.scale_bytes / 1024, 3),
                     table_kb=round(l.table_bytes / 1024, 3),
                     bias_kb=round(l.bias_bytes / 1024, 3),
                     total_kb=round(l.total_bytes / 1024, 3),
                     bits_per_weight=round(l.bits_per_weight, 3),
                     fp32_kb=round(l.fp32_weight_bytes / 1024, 3))
                for l in self.layers]


def fp32_reference_bytes(model: nn.Module) -> tuple[int, int, int]:
    """(param bytes, inference-relevant buffer bytes, param count) at fp32."""
    params = sum(p.numel() for p in model.parameters())
    buffers = 0
    for name, b in model.named_buffers():
        if name.endswith(("running_mean", "running_var")):
            buffers += b.numel()
    return params * 4, buffers * 4, params


@torch.no_grad()
def measure_model_size(model: nn.Module, weight_bits: int, act_bits: int,
                       group_size: int = 0, use_huffman: bool = True,
                       bias_bits: int = 16, scale_bits: int = 16,
                       reference_params: Optional[int] = None,
                       reference_buffer_bytes: Optional[int] = None,
                       keep_blobs: bool = False,
                       encodings: Optional[Dict[str, EncodedLayer]] = None
                       ) -> SizeReport:
    """Walk every quantized layer and total up the bytes actually needed.

    ``reference_params`` lets the caller supply the *original* (pre-folding)
    parameter count so the compression ratio is measured against the model the
    assignment started from, not against the already-slimmer folded graph.

    Note that ``group_size`` and ``scale_bits`` are accepted for a
    self-documenting call site but are *not* used to derive the scale cost:
    that comes from each layer's own ``scale_bytes``, so the accounting can
    never disagree with the quantizer that actually produced the scales.
    """
    report = SizeReport()
    param_bytes, buffer_bytes, n_params = fp32_reference_bytes(model)
    report.fp32_param_bytes = (reference_params * 4) if reference_params else param_bytes
    report.fp32_buffer_bytes = (reference_buffer_bytes if reference_buffer_bytes
                                is not None else buffer_bytes)
    report.n_params = reference_params or n_params

    for name, layer in quant_layers(model):
        codes = layer.integer_codes().detach().cpu().numpy()
        enc = encode_layer(codes, weight_bits, use_huffman=use_huffman,
                           keep_blobs=keep_blobs)
        if encodings is not None:
            encodings[name] = enc

        scale_bytes = layer.scale_bytes
        bias_bytes = (layer.bias.numel() * bias_bits // 8) if layer.bias is not None else 0
        kind = ("linear" if isinstance(layer, QuantLinear)
                else "depthwise" if layer.groups == layer.in_channels and layer.groups > 1
                else "pointwise" if layer.kernel_size == (1, 1) else "conv")

        report.layers.append(LayerSize(
            name=name, kind=kind, n_weights=int(codes.size), n_bits=weight_bits,
            sparsity=float((codes == 0).mean()), scheme=enc.scheme,
            payload_bits=int(enc.payload_bits), scale_bytes=int(scale_bytes),
            table_bytes=int(enc.table_bytes), bias_bytes=int(bias_bytes)))

    # Activation quantization parameters: one fp16 scale + one fp16 zero point
    # per observer.  Tiny, but charged anyway.
    n_obs = sum(1 for _ in act_quantizers(model))
    report.act_param_bytes = n_obs * 4

    # Any BatchNorm that survived (i.e. --no_fold_bn) still has to be shipped:
    # gamma, beta, running_mean and running_var, four fp16 values per channel.
    # Without this the un-folded configuration would look free, and the
    # BN-folding ablation would be meaningless.
    bn_values = sum(4 * m.num_features for m in model.modules()
                    if isinstance(m, nn.BatchNorm2d))
    report.bn_bytes = bn_values * 2
    return report


def format_size_table(report: SizeReport, max_rows: int = 0) -> str:
    """Human-readable summary printed by test.py / analyze.py."""
    d = report.to_dict()
    lines = [
        "=" * 78,
        "  MODEL SIZE ACCOUNTING (every byte a decoder needs)",
        "=" * 78,
        f"  FP32 reference (params only)     : {d['fp32_mb']:9.3f} MB "
        f"({report.n_params:,} params)",
        f"  + BatchNorm running stats (fp32) : {report.fp32_buffer_bytes / 1024:9.1f} KB"
        "   [folded away, excluded from ratios]",
        "-" * 78,
        f"  quantized weight payload         : {d['payload_kb']:9.1f} KB",
        f"  quantization scales              : {d['scale_kb']:9.1f} KB",
        f"  Huffman code books               : {d['huffman_table_kb']:9.1f} KB",
        f"  biases (fp16)                    : {d['bias_kb']:9.1f} KB",
        f"  per-layer headers                : {d['header_kb']:9.1f} KB",
        f"  activation scales/zero-points    : {d['act_param_kb']:9.1f} KB",
        f"  un-folded BatchNorm (fp16)       : {d['bn_kb']:9.1f} KB",
        "-" * 78,
        f"  metadata subtotal                : {d['metadata_kb']:9.1f} KB "
        f"({100 * d['metadata_fraction']:.1f}% of compressed model)",
        f"  COMPRESSED MODEL SIZE            : {d['compressed_mb']:9.3f} MB",
        "-" * 78,
        f"  mean bits / weight (incl. metadata): {d['avg_bits_per_weight']:7.3f}",
        f"  weight sparsity                  : {100 * d['overall_sparsity']:9.2f} %",
        f"  WEIGHT compression ratio         : {d['weight_compression_ratio']:9.2f} x",
        f"  MODEL  compression ratio         : {d['model_compression_ratio']:9.2f} x",
        "=" * 78,
    ]
    if max_rows:
        lines.append(f"{'layer':<40s}{'kind':<11s}{'#w':>9s}{'b/w':>7s}{'KB':>9s}")
        for row in report.to_rows()[:max_rows]:
            lines.append(f"{row['layer']:<40s}{row['kind']:<11s}{row['n_weights']:>9d}"
                         f"{row['bits_per_weight']:>7.2f}{row['total_kb']:>9.2f}")
    return "\n".join(lines)
