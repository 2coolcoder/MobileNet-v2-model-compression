"""Uniform quantization primitives -- written from scratch (no torch.ao / FBGEMM).

Two schemes are implemented:

* **Weights** -- *symmetric* uniform quantization.  Weight distributions in a
  trained network are close to zero-centred, so a symmetric grid wastes almost
  no code words and, more importantly, needs no zero-point (one fewer metadata
  tensor per layer, and integer arithmetic without cross terms).
  Scales are computed **per output channel** (or per group of ``group_size``
  weights inside a channel).  This matters enormously for MobileNet-v2: the
  depthwise convolutions have per-channel dynamic ranges that differ by two
  orders of magnitude, and a single per-tensor scale destroys the small ones.

* **Activations** -- *asymmetric* uniform quantization, because post-ReLU6
  tensors are one-sided ([0, 6]) and a symmetric grid would throw away half of
  the code words.

Both use an **MSE-optimal clipping search**: instead of taking the raw min/max
(which a single outlier can blow up), we scan candidate clipping ranges and
keep the one that minimises the squared reconstruction error.  Below 6 bits
this is worth several accuracy points.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Candidate clipping ratios scanned by the clipping search, as a fraction of
# max |x|.  The lower bound matters: an unrestricted search minimises the
# *mean* squared error of a group, which it can always do by clipping the
# single largest weight -- and that weight is usually the one the layer cares
# about most.  Restricting the range keeps the search useful without letting it
# over-clip.  See ``search_min_ratio`` in CompressionConfig.
_MSE_N_RATIOS = 41

# Error exponent for the clipping search.  p = 2 is plain MSE; p > 2 penalises
# the large individual errors that clipping produces, which matches what the
# network actually cares about far better at low bit-widths.
_MSE_DEFAULT_NORM = 2.4

# Smallest positive value fp16 can hold (a denormal).  Scales are stored as
# fp16, and a *pruned-away* group has max|w| = 0, so its scale would round to
# exactly zero -- and then 0/0 = NaN would poison the whole forward pass.
# Clamping to this floor is harmless: when every weight in a group is zero,
# any positive scale reproduces it exactly.
FP16_TINY = 5.960464477539063e-08


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def symmetric_qrange(n_bits: int) -> Tuple[int, int]:
    """Symmetric integer range, e.g. 8 bit -> [-127, 127].

    The range is deliberately symmetric (``-qmax`` rather than ``-qmax-1``) so
    that ``0`` is exactly representable and the code book is mirror-symmetric,
    which also helps the Huffman coder.
    """
    qmax = (1 << (n_bits - 1)) - 1
    return -qmax, qmax


def asymmetric_qrange(n_bits: int) -> Tuple[int, int]:
    """Unsigned integer range, e.g. 8 bit -> [0, 255]."""
    return 0, (1 << n_bits) - 1


def _grouped_view(weight: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int, int]:
    """Reshape ``weight`` to ``[out_channels, n_groups, group_size]``.

    Granularity is selected by ``group_size``:

    * ``-1`` -- one scale for the entire tensor (per-tensor, the coarsest and
      cheapest option; included so the ablation can quantify what per-channel
      scaling actually buys on MobileNet-v2's depthwise layers).
    * ``0``  -- one scale per output channel (the default).
    * ``N``  -- one scale per N weights inside a channel.

    Channels whose length is not a multiple of ``group_size`` are zero padded;
    the padding never affects a scale (it only adds zeros) and is dropped again
    by :func:`_expand_scales`.
    """
    if group_size == -1:                       # per-tensor
        return weight.reshape(1, 1, -1), weight.numel(), weight.numel()
    out_ch = weight.shape[0]
    flat = weight.reshape(out_ch, -1)
    k = flat.shape[1]
    if group_size and 0 < group_size < k:
        pad = (-k) % group_size
        if pad:
            flat = F.pad(flat, (0, pad))
        return flat.reshape(out_ch, -1, group_size), k, group_size
    return flat.reshape(out_ch, 1, k), k, k


def _expand_scales(scales: torch.Tensor, shape: torch.Size, k: int,
                   group_size: int) -> torch.Tensor:
    """Broadcast ``[out_ch, n_groups, 1]`` scales back to the weight shape."""
    if scales.shape[0] == 1 and scales.shape[1] == 1:       # per-tensor
        return scales.reshape(1).expand(int(torch.tensor(shape).prod())).reshape(shape)
    out_ch, n_groups = scales.shape[0], scales.shape[1]
    full = scales.expand(out_ch, n_groups, group_size).reshape(out_ch, -1)[:, :k]
    return full.reshape(shape)


def num_scales(weight: torch.Tensor, group_size: int) -> int:
    """How many scale values a layer stores -- needed by the size accounting."""
    if group_size == -1:
        return 1
    _, k, g = _grouped_view(weight, group_size)
    n_groups = (k + g - 1) // g
    return weight.shape[0] * n_groups


# --------------------------------------------------------------------------
# weight quantization
# --------------------------------------------------------------------------
@torch.no_grad()
def quantize_scales(scales: torch.Tensor, scale_bits: int,
                    eps: float = 1e-30) -> torch.Tensor:
    """Second-level quantization of the scales themselves ("double quantization").

    Fine-grained grouping is the cheapest way to buy accuracy at 3-4 bits, but
    at ``group_size=64`` the fp16 scales alone cost ~0.25 bits per weight --
    over 10% of the compressed model.  Scales are strictly positive and span
    several orders of magnitude, so a *logarithmic* grid fits them far better
    than a linear one: we store one fp16 offset and one fp16 step per layer,
    plus ``scale_bits`` per group.

    At ``scale_bits >= 16`` this is a no-op and scales are kept as raw fp16.
    """
    if scale_bits >= 16:
        return scales.half().float().clamp_min(FP16_TINY)
    log_s = torch.log2(scales.clamp_min(eps))
    lo, hi = log_s.min(), log_s.max()
    levels = (1 << scale_bits) - 1
    step = ((hi - lo) / levels).clamp_min(1e-12)
    lo, step = lo.half().float(), step.half().float()      # stored as fp16
    codes = torch.round((log_s - lo) / step).clamp(0, levels)
    return torch.exp2(lo + codes * step).clamp_min(FP16_TINY)


def scale_storage_bytes(n_scales: int, scale_bits: int) -> int:
    """Bytes needed for a layer's scales, including the fp16 offset/step pair."""
    if scale_bits >= 16:
        return n_scales * 2
    return (n_scales * scale_bits + 7) // 8 + 4


@torch.no_grad()
def compute_weight_scales(weight: torch.Tensor, n_bits: int, group_size: int = 0,
                          mse_search: bool = True, eps: float = 1e-12,
                          scale_bits: int = 16, min_ratio: float = 0.7,
                          norm: float = _MSE_DEFAULT_NORM) -> torch.Tensor:
    """Return per-channel / per-group symmetric scales, shaped like ``weight``.

    ``mse_search=False`` uses plain min/max (``scale = max|w| / qmax``).
    Otherwise the clipping ratio is searched over ``[min_ratio, 1.0]``,
    minimising ``sum |w - Q(w)|^norm`` per group.
    """
    groups, k, g = _grouped_view(weight, group_size)
    _, qmax = symmetric_qrange(n_bits)
    max_abs = groups.abs().amax(dim=-1, keepdim=True).clamp_min(eps)

    if not mse_search:
        scales = max_abs / qmax
    else:
        ratios = torch.linspace(min_ratio, 1.0, _MSE_N_RATIOS).to(
            weight.device, weight.dtype)
        best_err = None
        best_scale = None
        for ratio in ratios:
            scale = (max_abs * ratio / qmax).clamp_min(eps)
            q = torch.clamp(torch.round(groups / scale), -qmax, qmax)
            err = ((q * scale - groups).abs() ** norm).sum(dim=-1, keepdim=True)
            if best_err is None:
                best_err, best_scale = err, scale
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_scale = torch.where(better, scale, best_scale)
        scales = best_scale
    scales = quantize_scales(scales, scale_bits)
    return _expand_scales(scales, weight.shape, k, g)


@torch.no_grad()
def quantize_weight(weight: torch.Tensor, scales: torch.Tensor,
                    n_bits: int) -> torch.Tensor:
    """Real weights -> integer codes (still stored in a float tensor)."""
    qmin, qmax = symmetric_qrange(n_bits)
    return torch.clamp(torch.round(weight / scales.clamp_min(FP16_TINY)), qmin, qmax)


def dequantize_weight(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return codes * scales


class _RoundSTE(torch.autograd.Function):
    """round() with a straight-through gradient, used for QAT."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def fake_quantize_weight(weight: torch.Tensor, scales: torch.Tensor,
                         n_bits: int) -> torch.Tensor:
    """Differentiable quantize->dequantize (STE through the rounding)."""
    qmin, qmax = symmetric_qrange(n_bits)
    q = torch.clamp(_RoundSTE.apply(weight / scales.clamp_min(FP16_TINY)), qmin, qmax)
    return q * scales


# --------------------------------------------------------------------------
# activation quantization
# --------------------------------------------------------------------------
@torch.no_grad()
def asymmetric_qparams(x_min: torch.Tensor, x_max: torch.Tensor, n_bits: int,
                       eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale / zero-point for an asymmetric grid that always represents 0.

    Zero must be exactly representable, otherwise zero padding and pruned
    weights stop being free.
    """
    qmin, qmax = asymmetric_qrange(n_bits)
    x_min = torch.minimum(x_min, torch.zeros_like(x_min))
    x_max = torch.maximum(x_max, torch.zeros_like(x_max))
    scale = ((x_max - x_min) / (qmax - qmin)).clamp_min(eps)
    zero_point = torch.round(qmin - x_min / scale).clamp(qmin, qmax)
    return scale, zero_point


def fake_quantize_act(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                      n_bits: int) -> torch.Tensor:
    qmin, qmax = asymmetric_qrange(n_bits)
    q = torch.clamp(_RoundSTE.apply(x / scale) + zero_point, qmin, qmax)
    return (q - zero_point) * scale


@torch.no_grad()
def mse_optimal_range_from_hist(hist: torch.Tensor, bin_edges: torch.Tensor,
                                n_bits: int, n_candidates: int = 96
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick the activation clipping range that minimises quantization MSE.

    Works on the calibration histogram rather than on raw tensors so the search
    costs microseconds and needs no stored activations.  Values outside the
    candidate range are charged their full clipping error, which is what makes
    the criterion prefer tight ranges for heavy-tailed tensors and wide ranges
    for uniform ones.
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total_min, total_max = bin_edges[0], bin_edges[-1]
    best = None
    best_err = None
    for i in range(n_candidates):
        ratio = 1.0 - i / (n_candidates * 1.2)
        lo = total_min * ratio
        hi = total_max * ratio
        if hi - lo <= 0:
            continue
        scale, zp = asymmetric_qparams(lo.reshape(()), hi.reshape(()), n_bits)
        qmin, qmax = asymmetric_qrange(n_bits)
        q = torch.clamp(torch.round(centers / scale) + zp, qmin, qmax)
        deq = (q - zp) * scale
        err = (hist * (deq - centers) ** 2).sum()
        if best_err is None or err < best_err:
            best_err, best = err, (lo.clone(), hi.clone())
    if best is None:                                   # degenerate (all zeros)
        return total_min.clone(), total_max.clone()
    return best
