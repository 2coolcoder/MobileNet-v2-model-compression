# CS6886 - Assignment 2: MobileNet-v2 on CIFAR-10 + Custom Model Compression

Training MobileNet-v2 on CIFAR-10 from scratch, then compressing it with a
hand-written pipeline: **BatchNorm folding → sensitivity-guided magnitude
pruning → per-group symmetric weight quantization with min-max clipping →
log-domain double quantization of the scales → asymmetric activation
quantization → canonical Huffman/RLE entropy coding**, with byte-exact size
accounting that charges every metadata field.

No compression API or library function is used anywhere. `torch.ao.quantization`,
`torch.nn.utils.prune`, `bitsandbytes` and friends are absent from the
dependency list; the quantizers, the pruner, the Huffman coder and the size
model are all in `src/compression/`.

---

## Quick start

```bash
pip install -r requirements.txt
```

Reproduce everything end to end (≈2 h on one RTX 6000 Ada):

```bash
./scripts/run_all.sh
```

---

## Repository layout

```
train.py                     Q1  train the FP32 baseline
test.py                      Q2  compress + evaluate at one operating point
sweep.py                     Q3  grid over the compression knobs -> W&B
analyze.py                   Q4  self-tests, sensitivity, final point, figures

src/
  config.py                  paths, seeds, default hyper-parameters
  data.py                    CIFAR-10 transforms, loaders, Cutout, calibration split
  engine.py                  train/eval loops, warmup-cosine LR, param groups
  models/mobilenetv2.py      MobileNet-v2 adapted to 32x32 inputs
  activations.py             activation traffic + peak-buffer measurement
  plots.py                   every figure in the report
  compression/
    bnfold.py                fold BatchNorm into the preceding convolution
    quantize.py              uniform quant primitives, MSE clipping, scale quant
    modules.py               QuantConv2d / QuantLinear / ActQuant observers
    prune.py                 magnitude pruning + per-layer sensitivity analysis
    huffman.py               canonical Huffman + zero-run-length coding
    size.py                  byte-exact size accounting incl. all overheads
    pipeline.py              orchestration: convert -> prune -> calibrate -> QAT
```

---

## Commands

### Q1 : baseline

```bash
python train.py --epochs 200 --batch_size 128 --lr 0.1 --seed 42 --wandb
```

Writes `checkpoints/mobilenetv2_cifar10.pth`, `results/train_log.csv` and
`results/baseline_summary.json` (per-class accuracy + confusion matrix).

### Q2 : compression at one operating point

```bash
# the full method
python test.py --weight_quant_bits 4 --activation_quant_bits 8 \
               --group_size 0 --scale_bits 8 --prune_ratio 0.65 \
               --qat_epochs 12 --layer_table 20
```

### Q3 : sweep and the parallel-coordinates chart

```bash
python sweep.py                  # compute the grid: 108 configs, ~7 min
python upload_sweep.py           # create one W&B run per configuration
python sweep.py --dry_run        # just print the grid
```

sweep.py and upload_sweep.py are written separately because wandb logging 
slows down the script. You can also run `sweep.py --wandb`, which does the 
same thing.

Results land in `results/sweep_results.csv` and in the W&B project
[cs6886-a2-mobilenetv2-compression](https://wandb.ai/irupankarpodder-indian-institute-of-technology-madras/cs6886-a2-mobilenetv2-compression). The parallel-coordinates chart is built
from the columns `weight_quant_bits`, `activation_quant_bits`, `prune_ratio`,
`compression_ratio`, `model_size_mb`, `quantized_acc`; a local matplotlib twin
is written to `results/figures/parallel_coords.png`.

### Serializing the compressed model

```bash
python export.py        # writes checkpoints/mobilenetv2_compressed.bin
```

Writes a real binary artifact (packed Huffman streams, 8-bit scale codes, fp16
biases -- no pickle, no compression library), reports its size on disk against
the size `src/compression/size.py` predicts, and reloads it to confirm the
accuracy. On the reported operating point the file is 591,093 bytes against a
calculated 588,862.

### Q4 : final operating point

```bash
python analyze.py --sensitivity --target_sparsity 0.6 --tolerance 1.0
python analyze.py --final --weight_bits 4 --act_bits 8 --group_size 0 \
                  --scale_bits 8 --prune_ratio 0.65 --qat_epochs 12 \
                  --sparsity_map none --wandb
python analyze.py --figures
```

---

## Q5 : Reproducibility and seeds

* A single seed (**42**, `src/config.py:SEED`) drives everything.
  `src/utils.py:seed_all` seeds `random`, `numpy`, `torch` (CPU and all GPUs)
  and `PYTHONHASHSEED`; DataLoader shuffling uses an explicitly seeded
  `torch.Generator` and `worker_init_fn` re-seeds each worker deterministically.
