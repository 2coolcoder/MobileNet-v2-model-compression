"""Unstructured magnitude pruning with per-layer sensitivity allocation (Q2a).

Why pruning is part of the pipeline
-----------------------------------
Quantization alone is bounded: ``b`` bits per weight can never beat ``32/b``.
Sparsity attacks a different axis -- it removes weights entirely -- and it
composes with quantization because a pruned weight quantizes to the code word
``0``, which the Huffman stage then encodes in far fewer than ``b`` bits.  The
three stages multiply rather than add.

Why layers are treated differently
----------------------------------
MobileNet-v2 is already extremely parameter-efficient, and its layers are not
equally redundant:

* **Depthwise convolutions** hold only ``9 x C`` weights (~1.5% of the model)
  but every one of them carries a whole spatial filter.  Pruning them costs
  accuracy and saves almost nothing, so they are excluded.
* **The stem convolution** (864 weights) sees the raw image; excluded.
* **The classifier** (12.8k weights) is the only path to the logits; excluded.
* **Pointwise (1x1) convolutions** are ~95% of all parameters and are heavily
  over-parameterised -- this is where the sparsity budget is spent.

Within the prunable set, the per-layer threshold is chosen by a **sensitivity
analysis**: each layer is pruned alone at several sparsities and evaluated on a
held-out slice of the *training* set (never the test set).  Layers that
tolerate pruning receive a larger share of the global budget.

Why raw magnitude, on the *folded* weights
------------------------------------------
Pruning runs after BatchNorm folding, so the ranking sees ``|W * gamma /
sqrt(var + eps)|`` -- magnitude in the layer's **output** space rather than in
its raw parameter space.  That turns out to be exactly what we want.  A channel
whose BatchNorm gain is small genuinely contributes little to the next layer,
and folding makes that visible to a plain magnitude criterion for free.

A channel-RMS-normalized criterion (invariant to the folding scale) was
implemented and measured, and it is **worse** -- 86.6% versus 89.8% at 4 bits
and 50% sparsity -- precisely because it throws that signal away.  The simple
criterion wins here, and the experiment is why we know it.  It is kept behind
``prune_channel_normalized`` so the claim stays reproducible.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .modules import QuantConv2d, QuantLinear, quant_layers

SENSITIVITY_LEVELS = (0.3, 0.5, 0.7, 0.85, 0.95)


def is_depthwise(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d) and module.groups == module.in_channels \
        and module.groups > 1


def prunable_layers(model: nn.Module, prune_depthwise: bool = False,
                    prune_classifier: bool = False) -> List[Tuple[str, nn.Module]]:
    """Layers eligible for pruning, in forward order (see module docstring)."""
    layers = list(quant_layers(model))
    if not layers:
        return []
    first_name = layers[0][0]
    out = []
    for name, m in layers:
        if name == first_name:                      # stem conv
            continue
        if isinstance(m, QuantLinear) and not prune_classifier:
            continue
        if is_depthwise(m) and not prune_depthwise:
            continue
        out.append((name, m))
    return out


@torch.no_grad()
def clear_masks(model: nn.Module):
    for _, m in quant_layers(model):
        m.mask.fill_(1.0)
        m.calibrated = False


@torch.no_grad()
def importance(module: nn.Module, channel_normalized: bool = False) -> torch.Tensor:
    """Pruning criterion for one layer.

    Default is plain ``|w|`` on the folded weights (see the module docstring).
    ``channel_normalized=True`` divides by each output channel's RMS, making the
    ranking invariant to per-channel rescaling -- the measured-worse variant.
    """
    w = module.weight.detach()
    if not channel_normalized:
        return w.abs()
    flat = w.reshape(w.shape[0], -1)
    rms = flat.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    return (flat.abs() / rms).reshape(w.shape)


@torch.no_grad()
def apply_layer_mask(module: nn.Module, sparsity: float,
                     channel_normalized: bool = False):
    """Zero the ``sparsity`` fraction of least-important weights in one layer."""
    if sparsity <= 0:
        module.mask.fill_(1.0)
        module.calibrated = False
        return
    score = importance(module, channel_normalized)
    w = score.flatten()
    k = int(round(sparsity * w.numel()))
    k = min(max(k, 0), w.numel() - 1)
    if k == 0:
        module.mask.fill_(1.0)
    else:
        threshold = torch.kthvalue(w.float(), k).values
        module.mask.copy_((score > threshold).to(module.mask.dtype))
    module.calibrated = False


@torch.no_grad()
def global_magnitude_prune(model: nn.Module, ratio: float,
                           prune_depthwise: bool = False,
                           prune_classifier: bool = False,
                           per_layer: Optional[Dict[str, float]] = None,
                           channel_normalized: bool = False) -> Dict[str, float]:
    """Prune to a global sparsity ``ratio`` over the prunable weights.

    ``per_layer`` (from :func:`allocate_from_sensitivity`) overrides the single
    global threshold with per-layer sparsities; otherwise one threshold is
    computed over the concatenation of all prunable weights, which already
    allocates sparsity by magnitude distribution.
    """
    clear_masks(model)
    layers = prunable_layers(model, prune_depthwise, prune_classifier)
    if not layers or (ratio <= 0 and per_layer is None):
        return {name: 0.0 for name, _ in layers}

    if per_layer is not None:
        for name, m in layers:
            apply_layer_mask(m, per_layer.get(name, 0.0), channel_normalized)
    else:
        scores = {name: importance(m, channel_normalized) for name, m in layers}
        all_w = torch.cat([v.flatten() for v in scores.values()])
        k = int(round(ratio * all_w.numel()))
        k = min(max(k, 1), all_w.numel() - 1)
        threshold = torch.kthvalue(all_w.float(), k).values
        for name, m in layers:
            m.mask.copy_((scores[name] > threshold).to(m.mask.dtype))
            m.calibrated = False

    return {name: float(1.0 - m.mask.mean().item()) for name, m in layers}


@torch.no_grad()
def model_sparsity(model: nn.Module) -> float:
    """Overall fraction of zeroed weights across every quantized layer."""
    zeros = total = 0
    for _, m in quant_layers(model):
        zeros += int((m.mask == 0).sum().item())
        total += m.mask.numel()
    return zeros / max(total, 1)


@torch.no_grad()
def sensitivity_analysis(model: nn.Module, eval_fn, levels=SENSITIVITY_LEVELS,
                         prune_depthwise: bool = False,
                         verbose: bool = True) -> Dict[str, Dict[float, float]]:
    """Accuracy of the model when *one* layer at a time is pruned.

    ``eval_fn(model) -> accuracy`` is evaluated on the held-out training slice.
    Returns ``{layer_name: {sparsity: accuracy}}``.
    """
    clear_masks(model)
    baseline = eval_fn(model)
    results: Dict[str, Dict[float, float]] = {"__baseline__": {0.0: baseline}}
    for name, m in prunable_layers(model, prune_depthwise):
        results[name] = {}
        for level in levels:
            apply_layer_mask(m, level)
            results[name][level] = eval_fn(model)
        m.mask.fill_(1.0)
        m.calibrated = False
        if verbose:
            drops = " ".join(f"{lv:.2f}:{baseline - acc:5.2f}"
                             for lv, acc in results[name].items())
            print(f"  {name:<44s} drop@ {drops}", flush=True)
    return results


def allocate_from_sensitivity(sensitivity: Dict[str, Dict[float, float]],
                              model: nn.Module, target_ratio: float,
                              tolerance: float = 1.0,
                              prune_depthwise: bool = False) -> Dict[str, float]:
    """Turn the sensitivity curves into a per-layer sparsity assignment.

    For each layer we take the largest sparsity whose *solo* accuracy drop is
    below ``tolerance`` points; those per-layer caps are then scaled by a single
    global factor (bisection) so the weighted sparsity hits ``target_ratio``.
    """
    baseline = sensitivity["__baseline__"][0.0]
    layers = prunable_layers(model, prune_depthwise)
    sizes = {name: m.weight.numel() for name, m in layers}
    total = sum(sizes.values())

    caps: Dict[str, float] = {}
    for name, _ in layers:
        curve = sensitivity.get(name, {})
        allowed = [lv for lv, acc in sorted(curve.items()) if baseline - acc <= tolerance]
        caps[name] = max(allowed) if allowed else 0.0

    def achieved(alpha: float) -> float:
        return sum(min(caps[n] * alpha, 0.98) * sizes[n] for n, _ in layers) / total

    lo, hi = 0.0, 4.0
    if achieved(hi) < target_ratio:                 # budget unreachable: cap out
        return {n: min(caps[n] * hi, 0.98) for n, _ in layers}
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if achieved(mid) < target_ratio:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    return {n: float(min(caps[n] * alpha, 0.98)) for n, _ in layers}


def make_mask_hook(model: nn.Module):
    """Callable that re-applies every mask -- passed to ``train_one_epoch``."""
    layers = [m for _, m in quant_layers(model)]

    @torch.no_grad()
    def hook():
        for m in layers:
            m.apply_mask()

    return hook
