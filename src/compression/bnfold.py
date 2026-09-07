"""Fold BatchNorm into the preceding convolution (Q2b).

MobileNet-v2 has 52 BatchNorm layers holding 4 x C fp32 values each (weight,
bias, running_mean, running_var).  In the *uncompressed* model those are a
rounding error next to 2.2 M weights, but once the weights are quantized to 4
bits they would account for a large slice of the remaining bytes -- so folding
is not cosmetic, it directly improves the achievable compression ratio.

Folding is exact at inference time.  For a convolution followed by BN in eval
mode:

    y = gamma * (conv(x) - mu) / sqrt(var + eps) + beta
      = conv_{W', b'}(x)      with   W' = W * s,  b' = (b - mu) * s + beta
      and                            s  = gamma / sqrt(var + eps)   (per channel)

``s`` is per output channel, so the multiplication broadcasts over the weight's
trailing dimensions -- this works identically for dense, pointwise and
depthwise convolutions.
"""
import copy

import torch
import torch.nn as nn

from ..models.mobilenetv2 import ConvBNReLU


@torch.no_grad()
def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a single Conv2d (with bias) equivalent to ``conv`` then ``bn``."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      stride=conv.stride, padding=conv.padding,
                      dilation=conv.dilation, groups=conv.groups, bias=True)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)          # [out_ch]
    fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
    prev_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias.copy_((prev_bias - bn.running_mean) * scale + bn.bias)
    return fused.to(conv.weight.device)


@torch.no_grad()
def fold_bn(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Replace every ``ConvBNReLU`` block's conv+BN pair with a fused conv.

    The model must be in ``eval()`` mode -- folding uses the running statistics.
    Returns the folded model; ``inplace=False`` leaves the original untouched.
    """
    model = model if inplace else copy.deepcopy(model)
    model.eval()
    for module in model.modules():
        if isinstance(module, ConvBNReLU) and isinstance(module[1], nn.BatchNorm2d):
            module[0] = fuse_conv_bn(module[0], module[1])
            module[1] = nn.Identity()
    return model


@torch.no_grad()
def check_fold_equivalence(model: nn.Module, folded: nn.Module,
                           sample: torch.Tensor) -> float:
    """Max absolute logit difference between the original and folded models."""
    model.eval(), folded.eval()
    return (model(sample) - folded(sample)).abs().max().item()


def count_bn_parameters(model: nn.Module) -> int:
    """Number of fp32 values BatchNorm would cost if it were *not* folded
    (weight + bias + running_mean + running_var)."""
    total = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            total += 4 * module.num_features
    return total
