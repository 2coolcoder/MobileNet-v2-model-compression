"""Training / evaluation loops shared by every entry point."""
from typing import Optional

import torch
import torch.nn as nn

from .utils import AverageMeter


def _autocast(device: torch.device, enabled: bool):
    """bf16 autocast on CUDA (no GradScaler needed), no-op elsewhere."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, device,
                    amp: bool = True, scheduler=None,
                    mask_hook=None) -> tuple[float, float]:
    """One pass over ``loader``.

    ``mask_hook`` is called after each optimizer step; the pruning code uses it
    to re-apply binary masks so pruned weights stay at exactly zero during QAT.
    """
    model.train()
    loss_meter, correct, total = AverageMeter(), 0, 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if mask_hook is not None:
            mask_hook()
        if scheduler is not None:            # per-iteration schedulers
            scheduler.step()

        loss_meter.update(loss.item(), targets.size(0))
        correct += outputs.detach().argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return loss_meter.avg, 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion: Optional[nn.Module] = None,
             amp: bool = False) -> tuple[float, float]:
    """Top-1 accuracy (and mean loss when ``criterion`` is given).

    AMP defaults to *off* here: quantization experiments must not have their
    numerics perturbed by bf16 rounding on top of the fake-quant error.
    """
    model.eval()
    loss_meter, correct, total = AverageMeter(), 0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with _autocast(device, amp):
            outputs = model(inputs)
        if criterion is not None:
            loss_meter.update(criterion(outputs, targets).item(), targets.size(0))
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return loss_meter.avg, 100.0 * correct / max(total, 1)


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes: int = 10) -> torch.Tensor:
    """Row = ground truth, column = prediction.  Used for the Q1c failure-mode
    discussion."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
        preds = model(inputs).argmax(1).cpu()
        for t, p in zip(targets.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
    return cm


def build_param_groups(model, weight_decay: float):
    """Apply weight decay only to >=2-D parameters.

    Decaying BatchNorm scales/shifts and biases measurably hurts small-model
    CIFAR accuracy, so they are placed in a separate, decay-free group.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class WarmupCosineLR:
    """Linear warmup then cosine decay, stepped once per *iteration*."""

    def __init__(self, optimizer, base_lr: float, warmup_iters: int, total_iters: int,
                 min_lr: float = 0.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_iters = max(warmup_iters, 0)
        self.total_iters = max(total_iters, 1)
        self.min_lr = min_lr
        self.it = 0
        self.step(advance=False)

    def _lr_at(self, it: int) -> float:
        import math
        if it < self.warmup_iters:
            return self.base_lr * (it + 1) / max(self.warmup_iters, 1)
        progress = (it - self.warmup_iters) / max(self.total_iters - self.warmup_iters, 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self, advance: bool = True):
        lr = self._lr_at(self.it)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if advance:
            self.it += 1
        return lr

    @property
    def last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
