"""Q1 -- train the CIFAR-adapted MobileNet-v2 baseline.

Example
-------
    python train.py --epochs 200 --batch_size 128 --lr 0.1 --wandb
    python train.py --smoke                       # 2-epoch pipeline check
"""
import argparse
import json
import time

import torch
import torch.nn as nn

from src.config import (BASELINE_CKPT, CKPT_DIR, NUM_CLASSES, RESULTS_DIR, SEED,
                        TrainConfig, WANDB_PROJECT)
from src.data import get_cifar10
from src.engine import (WarmupCosineLR, build_param_groups, confusion_matrix,
                        evaluate, train_one_epoch)
from src.models.mobilenetv2 import mobilenet_v2
from src.utils import CSVLogger, count_parameters, dump_json, save_checkpoint, seed_all


def parse_args():
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="Train MobileNet-v2 on CIFAR-10")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--momentum", type=float, default=cfg.momentum)
    p.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    p.add_argument("--label_smoothing", type=float, default=cfg.label_smoothing)
    p.add_argument("--warmup_epochs", type=int, default=cfg.warmup_epochs)
    p.add_argument("--width_mult", type=float, default=cfg.width_mult)
    p.add_argument("--dropout", type=float, default=cfg.dropout)
    p.add_argument("--cutout_size", type=int, default=cfg.cutout_size)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--no_amp", action="store_true", help="disable bf16 autocast")
    p.add_argument("--out", type=str, default=str(BASELINE_CKPT))
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--run_name", type=str, default="baseline-mobilenetv2")
    p.add_argument("--smoke", action="store_true", help="2 epochs, tiny run")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs, args.warmup_epochs = 2, 0
        args.out = str(CKPT_DIR / "smoke_mobilenetv2.pth")

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
        cutout_size=args.cutout_size, download=False)

    model = mobilenet_v2(num_classes=NUM_CLASSES, width_mult=args.width_mult,
                         dropout=args.dropout).to(device, memory_format=torch.channels_last)
    n_params = count_parameters(model)
    print(f"MobileNet-v2 (width={args.width_mult}) | {n_params:,} params "
          f"| {n_params * 4 / 1024 ** 2:.2f} MB fp32", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(build_param_groups(model, args.weight_decay),
                                lr=args.lr, momentum=args.momentum, nesterov=True)
    iters_per_epoch = len(train_loader)
    scheduler = WarmupCosineLR(optimizer, args.lr,
                               warmup_iters=args.warmup_epochs * iters_per_epoch,
                               total_iters=args.epochs * iters_per_epoch)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=WANDB_PROJECT, name=args.run_name, job_type="train",
                         config={**vars(args), "n_params": n_params})

    logger = CSVLogger(RESULTS_DIR / "train_log.csv",
                       ["epoch", "lr", "train_loss", "train_acc", "test_loss",
                        "test_acc", "best_acc", "epoch_time_s"])
    best_acc, t_start = 0.0, time.time()

    for epoch in range(args.epochs):
        t0 = time.time()
        lr_now = scheduler.last_lr
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            amp=not args.no_amp, scheduler=scheduler)
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(args.out, model, epoch=epoch, test_acc=test_acc,
                            config=vars(args), n_params=n_params)

        row = dict(epoch=epoch + 1, lr=round(lr_now, 6),
                   train_loss=round(train_loss, 4), train_acc=round(train_acc, 3),
                   test_loss=round(test_loss, 4), test_acc=round(test_acc, 3),
                   best_acc=round(best_acc, 3), epoch_time_s=round(time.time() - t0, 1))
        logger.log(row)
        if run is not None:
            run.log(row, step=epoch + 1)
        print(f"Epoch {epoch + 1:3d}/{args.epochs} | lr {lr_now:.4f} "
              f"| train {train_loss:.4f}/{train_acc:.2f}% "
              f"| test {test_loss:.4f}/{test_acc:.2f}% | best {best_acc:.2f}% "
              f"| {row['epoch_time_s']}s", flush=True)

    # ---- final report artefacts (Q1c) ----------------------------------
    payload = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    _, final_acc = evaluate(model, test_loader, device)
    cm = confusion_matrix(model, test_loader, device, NUM_CLASSES)
    per_class = (cm.diag().float() / cm.sum(1).clamp_min(1).float() * 100).tolist()

    summary = dict(best_test_acc=best_acc, final_test_acc=final_acc,
                   n_params=n_params, fp32_size_mb=n_params * 4 / 1024 ** 2,
                   epochs=args.epochs, total_minutes=(time.time() - t_start) / 60,
                   per_class_acc=per_class, confusion_matrix=cm.tolist(),
                   config=vars(args))
    dump_json(RESULTS_DIR / "baseline_summary.json", summary)
    print(f"\nBaseline top-1: {final_acc:.2f}%  ({(time.time() - t_start) / 60:.1f} min)")
    print("Per-class acc:", " ".join(f"{a:.1f}" for a in per_class))

    if run is not None:
        run.summary.update({"best_test_acc": best_acc, "final_test_acc": final_acc})
        run.finish()


if __name__ == "__main__":
    main()
