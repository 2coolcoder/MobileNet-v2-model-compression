"""MobileNet-v2 adapted to 32x32 CIFAR inputs (Q1b).

Written from scratch following the reference architecture of Sandler et al.
(2018) and the layout used by ``torchvision.models.mobilenetv2`` so that the
module names line up with the familiar implementation, with two CIFAR
adaptations:

  1. the stem convolution uses stride 1 instead of stride 2, and
  2. the first ``t=6, c=24`` stage uses stride 1 instead of stride 2.

The stock ImageNet configuration downsamples 5 times (224 -> 7).  Applying it
to a 32x32 input would leave a 1x1 feature map before the classifier and
destroy all spatial information, so we remove two of the early strides and end
up with 32 -> 16 -> 8 -> 4, i.e. a 4x4 map feeding the global average pool.
"""
from typing import List, Optional

import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Round channel counts to a multiple of ``divisor`` (as in the paper's
    reference code) making sure we never round down by more than 10%."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class ConvBNReLU(nn.Sequential):
    """conv -> BatchNorm -> ReLU6, the basic building block of MobileNet-v2.

    Convolutions carry no bias because the following BatchNorm has one; after
    BN folding (see ``src/compression/bnfold.py``) the fused convolution gains
    an explicit bias term.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                 groups: int = 1, activation: bool = True):
        padding = (kernel_size - 1) // 2
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if activation:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """Expand (1x1) -> depthwise (3x3) -> project (1x1, *linear*) bottleneck.

    The projection has no activation ("linear bottleneck"): its output can be
    negative, which is why activation quantization has to treat these tensors
    as *signed* while post-ReLU6 tensors can use an unsigned range.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        hidden = int(round(in_ch * expand_ratio))
        self.use_res_connect = stride == 1 and in_ch == out_ch

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_ch, hidden, kernel_size=1))   # pointwise expand
        layers.extend([
            ConvBNReLU(hidden, hidden, kernel_size=3, stride=stride, groups=hidden),  # depthwise
            ConvBNReLU(hidden, out_ch, kernel_size=1, activation=False),              # linear project
        ])
        self.conv = nn.Sequential(*layers)
        # Quantization stub: replaced by an ActQuant observer during
        # compression so that the *block output* -- the linear-bottleneck
        # tensor, or the residual sum when there is a skip -- is quantized
        # exactly once.  It is an Identity in the uncompressed model.
        self.out_quant = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.conv(x) if self.use_res_connect else self.conv(x)
        return self.out_quant(out)


class MobileNetV2(nn.Module):
    """CIFAR MobileNet-v2.

    Args:
        num_classes: number of output logits (10 for CIFAR-10).
        width_mult: channel multiplier applied to every layer.
        dropout: dropout probability before the classifier.
        cifar_stem: if True use the 32x32 stride schedule described above.
    """

    #  t (expansion), c (out channels), n (repeats), s (stride of first repeat)
    #  The stride marked below is 2 in the ImageNet configuration.
    DEFAULT_SETTING = [
        [1, 16, 1, 1],
        [6, 24, 2, 1],   # ImageNet: stride 2 -- kept at 1 for 32x32 inputs
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0,
                 dropout: float = 0.2, cifar_stem: bool = True,
                 inverted_residual_setting: Optional[List[List[int]]] = None,
                 round_nearest: int = 8):
        super().__init__()
        setting = inverted_residual_setting or self.DEFAULT_SETTING
        if not cifar_stem:                        # restore the ImageNet strides
            setting = [list(s) for s in setting]
            setting[1][3] = 2

        input_channel = _make_divisible(32 * width_mult, round_nearest)
        last_channel = _make_divisible(1280 * max(1.0, width_mult), round_nearest)
        self.last_channel = last_channel

        stem_stride = 1 if cifar_stem else 2
        features: List[nn.Module] = [
            ConvBNReLU(3, input_channel, kernel_size=3, stride=stem_stride)]

        for t, c, n, s in setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel,
                                                 stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )
        # Quantization stubs for the network input and the pooled feature
        # vector (see InvertedResidual.out_quant).
        self.input_quant = nn.Identity()
        self.pool_quant = nn.Identity()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = self.pool_quant(torch.flatten(x, 1))
        return self.classifier(x)


def mobilenet_v2(num_classes: int = 10, width_mult: float = 1.0,
                 dropout: float = 0.2, cifar_stem: bool = True) -> MobileNetV2:
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult,
                       dropout=dropout, cifar_stem=cifar_stem)
