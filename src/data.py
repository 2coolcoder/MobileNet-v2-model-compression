"""CIFAR-10 input pipeline: normalization, augmentation and loaders (Q1a)."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import CIFAR10_MEAN, CIFAR10_STD, DATA_DIR, SEED
from .utils import worker_init_fn


class Cutout:
    """Random square erasure, applied on the *normalized* tensor.

    Implemented here rather than pulled from a library so the augmentation
    stack is fully specified in this repo (DeVries & Taylor, 2017).
    """

    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip([y - self.length // 2, y + self.length // 2], 0, h)
            x1, x2 = np.clip([x - self.length // 2, x + self.length // 2], 0, w)
            mask[y1:y2, x1:x2] = 0.0
        return img * torch.from_numpy(mask).expand_as(img)

    def __repr__(self):
        return f"Cutout(n_holes={self.n_holes}, length={self.length})"


def build_transforms(train: bool, cutout_holes: int = 1, cutout_size: int = 16):
    """Return the exact transform pipeline used for train / eval."""
    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    tfms = [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    if cutout_holes > 0 and cutout_size > 0:
        tfms.append(Cutout(cutout_holes, cutout_size))
    return transforms.Compose(tfms)


def get_datasets(cutout_holes: int = 1, cutout_size: int = 16, download: bool = True):
    train_set = datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=download,
        transform=build_transforms(True, cutout_holes, cutout_size))
    test_set = datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=download,
        transform=build_transforms(False))
    return train_set, test_set


def get_cifar10(batch_size: int = 128, num_workers: int = 8, seed: int = SEED,
                cutout_holes: int = 1, cutout_size: int = 16,
                download: bool = True, eval_batch_size: int | None = None):
    """Standard train/test loaders.  Shuffling is seeded for reproducibility."""
    train_set, test_set = get_datasets(cutout_holes, cutout_size, download)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=False, generator=generator,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0)
    test_loader = DataLoader(
        test_set, batch_size=eval_batch_size or max(batch_size, 256),
        shuffle=False, num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn, persistent_workers=num_workers > 0)
    return train_loader, test_loader


def get_calibration_loader(batch_size: int = 128, num_batches: int = 32,
                           num_workers: int = 4, seed: int = SEED):
    """Deterministic subset of the *training* set used to calibrate activation
    ranges and to run the pruning sensitivity sweep.

    Augmentation is disabled: calibration must see the same statistics the
    model sees at inference time.
    """
    calib_set = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False,
                                 transform=build_transforms(False))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(calib_set))[: batch_size * num_batches]
    loader = DataLoader(Subset(calib_set, idx.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader


def get_heldout_loader(n: int = 2000, batch_size: int = 256, num_workers: int = 4,
                       seed: int = SEED):
    """Held-out slice of the training set (never used for calibration) that
    drives the per-layer pruning sensitivity analysis, so that the CIFAR-10
    test set is only ever touched for final reporting.
    """
    heldout = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False,
                               transform=build_transforms(False))
    rng = np.random.RandomState(seed + 1)
    idx = rng.permutation(len(heldout))[-n:]
    return DataLoader(Subset(heldout, idx.tolist()), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers, pin_memory=True)
