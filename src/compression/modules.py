"""Quantization-aware replacements for Conv2d / Linear / activations (Q2a, Q2b).

Design notes
------------
* ``QuantConv2d`` / ``QuantLinear`` are *simulated* (fake) quantization: the
  weight is quantized to an integer grid and immediately dequantized, so the
  convolution still runs in fp32 on the GPU while producing exactly the values
  an integer kernel would.  This is the standard way to measure PTQ/QAT
  accuracy without writing custom CUDA kernels, and the reported model size is
  computed from the *integer* representation (see ``size.py``), never from the
  simulated fp32 tensor.
* A binary ``mask`` buffer implements pruning.  It is applied *before*
  quantization so that zeros survive quantization exactly (the symmetric grid
  always contains 0) and so the Huffman coder sees the real run lengths.
* ``ActQuant`` is a four-state observer/quantizer.  Calibration needs two
  passes: pass 1 records the global min/max, pass 2 fills a histogram inside
  that range, from which the MSE-optimal clipping range is derived.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import (asymmetric_qparams, compute_weight_scales, fake_quantize_act,
                       fake_quantize_weight, mse_optimal_range_from_hist,
                       num_scales, quantize_weight, scale_storage_bytes)

# Observer/quantizer states for ActQuant.
OFF, MINMAX, HIST, QUANT = "off", "minmax", "hist", "quant"


class _QuantWeightMixin:
    """Shared weight-quantization machinery for conv and linear layers."""

    def _init_quant(self, n_bits: int, group_size: int, mse_search: bool,
                    scale_bits: int = 16, search_min_ratio: float = 0.7,
                    search_norm: float = 2.4):
        self.w_bits = n_bits
        self.group_size = group_size
        self.mse_search = mse_search
        self.scale_bits = scale_bits
        self.search_min_ratio = search_min_ratio
        self.search_norm = search_norm
        self.weight_quant_enabled = True
        self.register_buffer("w_scale", torch.ones_like(self.weight))
        self.register_buffer("mask", torch.ones_like(self.weight))
        self.calibrated = False

    @torch.no_grad()
    def calibrate_weight(self):
        """Compute (and cache) the weight scales from the current weights."""
        w = self.weight * self.mask
        self.w_scale.copy_(compute_weight_scales(
            w, self.w_bits, self.group_size, self.mse_search,
            scale_bits=self.scale_bits, min_ratio=self.search_min_ratio,
            norm=self.search_norm))
        self.calibrated = True

    def effective_weight(self) -> torch.Tensor:
        """Masked weight, fake-quantized when quantization is enabled."""
        w = self.weight * self.mask
        if not self.weight_quant_enabled:
            return w
        if not self.calibrated:
            self.calibrate_weight()
        return fake_quantize_weight(w, self.w_scale, self.w_bits)

    @torch.no_grad()
    def integer_codes(self) -> torch.Tensor:
        """Integer weight codes actually stored on disk (used by size.py)."""
        if not self.calibrated:
            self.calibrate_weight()
        return quantize_weight(self.weight * self.mask, self.w_scale, self.w_bits)

    @torch.no_grad()
    def apply_mask(self):
        """Re-zero pruned weights (called after every optimizer step in QAT)."""
        self.weight.mul_(self.mask)

    @property
    def n_weight_scales(self) -> int:
        return num_scales(self.weight, self.group_size)

    @property
    def scale_bytes(self) -> int:
        """Bytes this layer spends on scales, including double-quantization
        offset/step when ``scale_bits < 16``."""
        return scale_storage_bytes(self.n_weight_scales, self.scale_bits)

    @torch.no_grad()
    def round_bias_to_fp16(self):
        """Store biases at fp16 -- and *simulate* that rounding, so the accuracy
        we report is the accuracy of the model we actually charge for."""
        if self.bias is not None:
            self.bias.data.copy_(self.bias.data.half().float())

    @property
    def sparsity(self) -> float:
        return 1.0 - self.mask.mean().item()


class QuantConv2d(nn.Conv2d, _QuantWeightMixin):
    """Conv2d with per-channel / per-group symmetric weight quantization."""

    def __init__(self, *args, n_bits: int = 8, group_size: int = 0,
                 mse_search: bool = True, scale_bits: int = 16,
                 search_min_ratio: float = 0.7, search_norm: float = 2.4, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quant(n_bits, group_size, mse_search, scale_bits,
                         search_min_ratio, search_norm)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, n_bits: int = 8, group_size: int = 0,
                  mse_search: bool = True, scale_bits: int = 16,
                  search_min_ratio: float = 0.7,
                  search_norm: float = 2.4) -> "QuantConv2d":
        q = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                groups=conv.groups, bias=conv.bias is not None,
                n_bits=n_bits, group_size=group_size, mse_search=mse_search,
                scale_bits=scale_bits, search_min_ratio=search_min_ratio,
                search_norm=search_norm)
        q.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            q.bias.data.copy_(conv.bias.data)
        return q.to(conv.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(x, self.effective_weight(), self.bias)


class QuantLinear(nn.Linear, _QuantWeightMixin):
    """Linear layer with the same weight-quantization scheme."""

    def __init__(self, *args, n_bits: int = 8, group_size: int = 0,
                 mse_search: bool = True, scale_bits: int = 16,
                 search_min_ratio: float = 0.7, search_norm: float = 2.4, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quant(n_bits, group_size, mse_search, scale_bits,
                         search_min_ratio, search_norm)

    @classmethod
    def from_linear(cls, lin: nn.Linear, n_bits: int = 8, group_size: int = 0,
                    mse_search: bool = True, scale_bits: int = 16,
                    search_min_ratio: float = 0.7,
                    search_norm: float = 2.4) -> "QuantLinear":
        q = cls(lin.in_features, lin.out_features, bias=lin.bias is not None,
                n_bits=n_bits, group_size=group_size, mse_search=mse_search,
                scale_bits=scale_bits, search_min_ratio=search_min_ratio,
                search_norm=search_norm)
        q.weight.data.copy_(lin.weight.data)
        if lin.bias is not None:
            q.bias.data.copy_(lin.bias.data)
        return q.to(lin.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


class ActQuant(nn.Module):
    """Per-tensor asymmetric activation observer + fake-quantizer.

    ``signed`` only changes the *expected* sign of the data; the grid is
    asymmetric either way, so a post-ReLU6 tensor automatically ends up with
    ``min = 0`` and spends all its code words on [0, 6].

    ``tag`` records where in the network this observer sits, which is what the
    activation-compression accounting in ``src/activations.py`` reports on.
    """

    N_BINS = 2048

    def __init__(self, n_bits: int = 8, signed: bool = False, tag: str = ""):
        super().__init__()
        self.n_bits = n_bits
        self.signed = signed
        self.tag = tag
        self.state = OFF
        self.enabled = True
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("hist", torch.zeros(self.N_BINS))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        # ``state`` is a plain attribute, so it does not survive a state_dict
        # round trip.  ``frozen`` is a buffer that does: it records that
        # calibration finished, and the load hook below restores the QUANT
        # state from it.  Without this a reloaded checkpoint would silently
        # skip activation quantization and report the wrong accuracy.
        self.register_buffer("frozen", torch.tensor(0.0))
        self.numel_per_sample = 0          # filled in by the activation profiler
        self.shape: Optional[tuple] = None

    # ------------------------------------------------------------ calib ----
    @torch.no_grad()
    def _observe_minmax(self, x: torch.Tensor):
        self.min_val = torch.minimum(self.min_val, x.min().float())
        self.max_val = torch.maximum(self.max_val, x.max().float())

    @torch.no_grad()
    def _observe_hist(self, x: torch.Tensor):
        lo, hi = self.min_val.item(), self.max_val.item()
        if hi <= lo:
            return
        self.hist += torch.histc(x.detach().float().flatten(), bins=self.N_BINS,
                                 min=lo, max=hi).to(self.hist.device)

    @torch.no_grad()
    def finalize(self, mse_search: bool = True):
        """Turn the collected statistics into (scale, zero_point)."""
        lo, hi = self.min_val.clone(), self.max_val.clone()
        if mse_search and self.hist.sum() > 0:
            edges = torch.linspace(lo.item(), hi.item(), self.N_BINS + 1,
                                   device=self.hist.device)
            lo, hi = mse_optimal_range_from_hist(self.hist, edges, self.n_bits)
        scale, zp = asymmetric_qparams(lo.reshape(()), hi.reshape(()), self.n_bits)
        self.scale.copy_(scale.to(self.scale.device))
        self.zero_point.copy_(zp.to(self.zero_point.device))
        self.frozen.fill_(1.0)
        self.state = QUANT

    # ---------------------------------------------------------- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shape is None:
            self.shape = tuple(x.shape[1:])
            self.numel_per_sample = int(torch.tensor(self.shape).prod().item())
        if self.state == MINMAX:
            self._observe_minmax(x)
            return x
        if self.state == HIST:
            self._observe_hist(x)
            return x
        if self.state == QUANT and self.enabled:
            return fake_quantize_act(x, self.scale, self.zero_point, self.n_bits)
        return x

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if float(self.frozen.item()) > 0.5:
            self.state = QUANT

    def extra_repr(self) -> str:
        return f"bits={self.n_bits}, signed={self.signed}, tag={self.tag}, state={self.state}"


def set_act_state(model: nn.Module, state: str):
    for m in model.modules():
        if isinstance(m, ActQuant):
            m.state = state


def finalize_act(model: nn.Module, mse_search: bool = True):
    for m in model.modules():
        if isinstance(m, ActQuant):
            m.finalize(mse_search)


def set_weight_quant(model: nn.Module, enabled: bool):
    for m in model.modules():
        if isinstance(m, _QuantWeightMixin):
            m.weight_quant_enabled = enabled


def quant_layers(model: nn.Module):
    """Yield ``(name, module)`` for every quantized weight layer."""
    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            yield name, m


def act_quantizers(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, ActQuant):
            yield name, m
