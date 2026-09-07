"""Central configuration: paths, seeds and default hyper-parameters.

Every entry point (train.py / test.py / sweep.py / analyze.py) reads its
defaults from here so that the numbers quoted in REPORT.md can be traced back
to a single place.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BASELINE_CKPT = CKPT_DIR / "mobilenetv2_cifar10.pth"

# ------------------------------------------------------------ constants ----
SEED = 42
NUM_CLASSES = 10
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# CIFAR-10 channel statistics computed over the 50k training images.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

WANDB_PROJECT = "cs6886-a2-mobilenetv2-compression"


@dataclass
class TrainConfig:
    """Q1 baseline training recipe."""
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    cutout_holes: int = 1
    cutout_size: int = 16
    width_mult: float = 1.0
    dropout: float = 0.2
    num_workers: int = 8
    amp: bool = True
    seed: int = SEED


@dataclass
class CompressionConfig:
    """Q2/Q3/Q4 compression recipe.  Every field is exposed on the CLI."""
    weight_bits: int = 8            # bit-width of quantized weights
    act_bits: int = 8               # bit-width of quantized activations
    group_size: int = 0             # 0 -> per-output-channel, >0 -> per-group
    prune_ratio: float = 0.0        # global sparsity target on prunable layers
    prune_channel_normalized: bool = False  # rank |w|/rms(channel) instead of |w|
    huffman: bool = True            # entropy-code the quantized index stream
    fold_bn: bool = True            # fold BatchNorm into the preceding conv
    calib_batches: int = 32         # training batches used for act calibration
    mse_search: bool = False        # clipping search (measured to hurt -- see REPORT)
    search_min_ratio: float = 0.7   # lowest clipping ratio the search may pick
    search_norm: float = 2.4        # error exponent used by the clipping search
    bias_bits: int = 16             # biases stored in fp16
    scale_bits: int = 16            # quantization scales stored in fp16
    qat_epochs: int = 0             # >0 -> quantization-aware fine-tuning
    qat_lr: float = 1e-3
    seed: int = SEED

    def as_dict(self):
        return asdict(self)
