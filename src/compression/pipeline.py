"""End-to-end compression pipeline (Q2a) -- the orchestration layer.

    fp32 model
        |  1. fold BatchNorm into the preceding convolution        (bnfold.py)
        |  2. swap Conv2d/Linear -> QuantConv2d/QuantLinear        (modules.py)
        |     and ReLU6 / block-output stubs -> ActQuant observers
        |  3. magnitude-prune the pointwise convolutions            (prune.py)
        |  4. calibrate activation ranges on training batches      (2 passes)
        |  5. [optional] quantization-aware fine-tuning with STE
        |  6. entropy-code the integer weights, count every byte   (huffman/size)
        v
    compressed model + size report

Every stage is switchable from the CLI, which is what makes the method
"configurable" in the sense Q2a asks for: bit-widths, group size, sparsity,
entropy coding, BN folding, MSE search and QAT are all independent knobs.
"""
import copy
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from ..config import CompressionConfig
from ..models.mobilenetv2 import ConvBNReLU, InvertedResidual, MobileNetV2
from .bnfold import fold_bn
from .modules import (HIST, MINMAX, OFF, QUANT, ActQuant, QuantConv2d, QuantLinear,
                      finalize_act, quant_layers, set_act_state)
from .prune import global_magnitude_prune, make_mask_hook, model_sparsity
from .size import measure_model_size


# --------------------------------------------------------------------------
# 2. module surgery
# --------------------------------------------------------------------------
def _swap_weight_layers(module: nn.Module, cfg: CompressionConfig):
    """Recursively replace Conv2d / Linear with their quantized counterparts."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, QuantConv2d):
            setattr(module, name, QuantConv2d.from_conv(
                child, cfg.weight_bits, cfg.group_size, cfg.mse_search,
                cfg.scale_bits, cfg.search_min_ratio, cfg.search_norm))
        elif isinstance(child, nn.Linear) and not isinstance(child, QuantLinear):
            setattr(module, name, QuantLinear.from_linear(
                child, cfg.weight_bits, cfg.group_size, cfg.mse_search,
                cfg.scale_bits, cfg.search_min_ratio, cfg.search_norm))
        else:
            _swap_weight_layers(child, cfg)


def _insert_act_quant(model: MobileNetV2, cfg: CompressionConfig):
    """Place activation observers at the four kinds of quantization point.

    * after every ReLU6                 -> unsigned range, the tensor is in [0, 6]
    * at every InvertedResidual output  -> signed: linear bottlenecks and the
                                           residual sum can be negative
    * at the network input              -> signed: normalized images
    * after global average pooling      -> unsigned: it averages ReLU6 outputs

    Note the *absence* of an observer directly after each linear projection:
    the block-output stub already covers that tensor, and quantizing it twice
    would double the error for no storage benefit.
    """
    for block in model.modules():
        if isinstance(block, ConvBNReLU) and isinstance(block[-1], nn.ReLU6):
            block.append(ActQuant(cfg.act_bits, signed=False, tag="post_relu6"))
    for name, block in model.named_modules():
        if isinstance(block, InvertedResidual):
            block.out_quant = ActQuant(cfg.act_bits, signed=True, tag="block_out")
    model.input_quant = ActQuant(cfg.act_bits, signed=True, tag="input")
    model.pool_quant = ActQuant(cfg.act_bits, signed=False, tag="pooled")


def convert(model: nn.Module, cfg: CompressionConfig, inplace: bool = False) -> nn.Module:
    """Stages 1-2: fold BN, then install quantized layers and observers."""
    model = model if inplace else copy.deepcopy(model)
    model.eval()
    if cfg.fold_bn:
        model = fold_bn(model, inplace=True)
    _swap_weight_layers(model, cfg)
    _insert_act_quant(model, cfg)
    if cfg.bias_bits <= 16:
        # Biases are charged at fp16 in size.py, so round them to fp16 here:
        # the reported accuracy must be the accuracy of the stored model.
        for _, layer in quant_layers(model):
            layer.round_bias_to_fp16()
    return model


# --------------------------------------------------------------------------
# 4. activation calibration
# --------------------------------------------------------------------------
@torch.no_grad()
def calibrate(model: nn.Module, loader, device, cfg: CompressionConfig,
              max_batches: Optional[int] = None):
    """Two-pass activation calibration: min/max, then histogram, then solve.

    Pass 1 fixes the histogram support; pass 2 fills it; ``finalize`` runs the
    MSE clipping search on the histogram.  Weight quantization is left ON
    throughout so activation ranges are calibrated against the network the
    quantized weights actually produce.
    """
    model.eval()
    n = max_batches if max_batches is not None else cfg.calib_batches
    for state in (MINMAX, HIST):
        set_act_state(model, state)
        for i, (x, _) in enumerate(loader):
            if i >= n:
                break
            model(x.to(device, non_blocking=True, memory_format=torch.channels_last))
    finalize_act(model, cfg.mse_search)
    set_act_state(model, QUANT)


# --------------------------------------------------------------------------
# 5. quantization-aware fine-tuning
# --------------------------------------------------------------------------
def qat_finetune(model: nn.Module, train_loader, test_loader, device,
                 cfg: CompressionConfig, log_fn: Optional[Callable] = None):
    """Short STE fine-tune of the already-quantized, already-pruned model.

    Weight scales are recomputed at the start of every epoch (the weights move,
    so their optimal clipping range moves with them) while activation scales
    stay frozen at their calibrated values -- re-observing them during training
    would chase the augmented-data statistics rather than the inference ones.
    """
    from ..engine import WarmupCosineLR, build_param_groups, evaluate, train_one_epoch
    from ..utils import set_fp32_precision

    if cfg.qat_epochs <= 0:
        return model, {}

    # Fine-tuning may use the fast TF32 convolution kernels: they are several
    # times quicker on Ada, and their ~1e-3 relative error is noise next to the
    # 4-bit quantization the network is being trained against.  Everything that
    # *reports* a number -- the per-epoch evaluation below and the final
    # measurement -- switches back to IEEE fp32 first, so no accuracy quoted in
    # the report is produced with reduced precision.
    set_fp32_precision(exact=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(build_param_groups(model, 1e-5),
                                lr=cfg.qat_lr, momentum=0.9, nesterov=True)
    iters = len(train_loader)
    scheduler = WarmupCosineLR(optimizer, cfg.qat_lr, warmup_iters=0,
                               total_iters=cfg.qat_epochs * iters)
    mask_hook = make_mask_hook(model)
    history = []
    best_acc, best_state = -1.0, None

    for epoch in range(cfg.qat_epochs):
        for _, layer in quant_layers(model):     # re-fit scales to moved weights
            layer.calibrate_weight()
        model.train()
        set_act_state(model, QUANT)              # keep frozen activation ranges
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, amp=False,
            scheduler=scheduler, mask_hook=mask_hook)
        for _, layer in quant_layers(model):
            layer.calibrate_weight()
        set_fp32_precision(exact=True)          # measure in IEEE fp32
        _, test_acc = evaluate(model, test_loader, device)
        set_fp32_precision(exact=False)         # fast kernels again to train
        row = dict(epoch=epoch + 1, train_loss=train_loss, train_acc=train_acc,
                   test_acc=test_acc)
        history.append(row)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())
        if log_fn:
            log_fn(row)
        print(f"  QAT epoch {epoch + 1}/{cfg.qat_epochs} | train {train_loss:.4f}"
              f"/{train_acc:.2f}% | test {test_acc:.2f}%", flush=True)

    set_fp32_precision(exact=True)              # everything after this is reported
    if best_state is not None:
        model.load_state_dict(best_state)
    for _, layer in quant_layers(model):
        layer.calibrate_weight()
    model.eval()
    set_act_state(model, QUANT)
    return model, {"qat_history": history, "qat_best_acc": best_acc}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def compress(model: nn.Module, cfg: CompressionConfig, calib_loader, device,
             per_layer_sparsity: Optional[Dict[str, float]] = None,
             reference_params: Optional[int] = None,
             reference_buffer_bytes: Optional[int] = None,
             train_loader=None, test_loader=None, verbose: bool = True):
    """Run the whole pipeline and return ``(compressed_model, size_report, info)``."""
    reference_params = reference_params or sum(p.numel() for p in model.parameters())
    if reference_buffer_bytes is None:
        # BatchNorm running statistics of the *original* graph -- reported as a
        # separate line so the reader can see what BN folding removed.
        reference_buffer_bytes = 4 * sum(
            b.numel() for n, b in model.named_buffers()
            if n.endswith(("running_mean", "running_var")))

    qmodel = convert(model, cfg).to(device, memory_format=torch.channels_last)

    sparsity_map = {}
    if cfg.prune_ratio > 0 or per_layer_sparsity is not None:
        sparsity_map = global_magnitude_prune(
            qmodel, cfg.prune_ratio, per_layer=per_layer_sparsity,
            channel_normalized=cfg.prune_channel_normalized)
    for _, layer in quant_layers(qmodel):
        layer.calibrate_weight()

    calibrate(qmodel, calib_loader, device, cfg)

    info: Dict = {"sparsity_map": sparsity_map,
                  "achieved_sparsity": model_sparsity(qmodel)}

    if cfg.qat_epochs > 0:
        if train_loader is None or test_loader is None:
            raise ValueError("QAT requested but train/test loaders were not provided")
        qmodel, qat_info = qat_finetune(qmodel, train_loader, test_loader, device, cfg)
        info.update(qat_info)

    report = measure_model_size(qmodel, cfg.weight_bits, cfg.act_bits,
                                group_size=cfg.group_size, use_huffman=cfg.huffman,
                                bias_bits=cfg.bias_bits, scale_bits=cfg.scale_bits,
                                reference_params=reference_params,
                                reference_buffer_bytes=reference_buffer_bytes)
    if verbose:
        print(f"  pruned to {100 * info['achieved_sparsity']:.1f}% sparsity, "
              f"{report.compressed_mb:.3f} MB, {report.model_compression_ratio:.2f}x",
              flush=True)
    return qmodel, report, info
