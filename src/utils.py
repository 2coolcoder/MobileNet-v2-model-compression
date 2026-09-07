"""Small shared helpers: seeding, metric meters, checkpoint and CSV io."""
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_fp32_precision(exact: bool = True):
    """Turn TF32 off (``exact=True``) for convolutions and matmuls.

    Ada/Ampere GPUs silently run fp32 convolutions in TF32, which carries ~1e-3
    relative error.  That is harmless for training but *not* for this
    assignment: it is the same order as the error introduced by 8-bit
    quantization, so every evaluation and compression path pins IEEE fp32 and
    only the baseline training run (which uses bf16 autocast anyway) keeps the
    fast kernels.
    """
    mode = "ieee" if exact else "tf32"
    try:                                     # torch >= 2.9 API
        torch.backends.cudnn.conv.fp32_precision = mode
        torch.backends.cuda.matmul.fp32_precision = mode
    except AttributeError:                   # older releases
        torch.backends.cudnn.allow_tf32 = not exact
        torch.backends.cuda.matmul.allow_tf32 = not exact


def seed_all(seed: int = 42, deterministic: bool = False):
    """Seed python / numpy / torch (CPU + all GPUs).

    ``deterministic=True`` additionally disables the cuDNN autotuner and asks
    for deterministic kernels.  We keep it *off* for the 200-epoch baseline
    (benchmark mode is ~1.3x faster) and document that choice in the README;
    all compression/evaluation code paths are deterministic regardless because
    they run under ``torch.no_grad`` with fixed data order.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int):
    """Deterministic per-worker seeding for the DataLoader."""
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


class AverageMeter:
    """Running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0


class CSVLogger:
    """Append-only CSV writer that creates the header on first use."""

    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.fieldnames).writerow(
                {k: row.get(k, "") for k in self.fieldnames})


def save_checkpoint(path, model, **extra):
    payload = {"state_dict": model.state_dict()}
    payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model, device="cpu", strict: bool = True):
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=strict)
    return payload


def dump_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def human_mb(num_bytes: float) -> float:
    return num_bytes / (1024.0 ** 2)
