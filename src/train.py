"""Train a model and log the run.

Each run appends one row to results/runs.csv, writes its per-epoch curves to
results/history/<run_id>.csv and saves the best weights to results/checkpoints/.
Only validation scores are reported here; the test set is for the final evaluation.

Usage (from the project root):
    python -m src.train                                   # MLP, Adam, defaults
    python -m src.train --optimizer sgd --lr 0.01 --batch-size 64
"""

import argparse
import csv
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src import config
from src.dataset import crop_splits, landmark_splits, to_loader
from src.models import CNN, MLP, count_params

RUN_FIELDS = [
    "run_id", "model", "optimizer", "lr", "momentum", "batch_size", "dropout", "scheduler",
    "max_epochs", "patience", "epochs_run", "best_epoch", "train_loss", "val_loss", "val_acc", "val_f1",
    "params", "train_seconds", "split", "include_enhanced", "mirror_train", "seed", "notes",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD only")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--scheduler", choices=["none", "plateau"], default="none")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5, help="stop after this many epochs without val loss improving")
    p.add_argument("--split", choices=["block", "random"], default="block")
    p.add_argument("--no-enhanced", action="store_true", help="drop brightening-recovered samples from train")
    p.add_argument("--no-mirror", action="store_true", help="no mirror augmentation")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--notes", default="")
    p.add_argument("--no-log", action="store_true", help="test run: don't write runs.csv, history or checkpoint")
    return p.parse_args()


def make_optimizer(args, params):
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr)
    return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total_loss / len(targets),
        "acc": float((preds == targets).mean()),
        "f1": f1_score(targets, preds, average="macro"),
    }


def append_csv(path, fields, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{args.model}_{args.optimizer}_bs{args.batch_size}"

    load_splits = landmark_splits if args.model == "mlp" else crop_splits
    splits = load_splits(args.split, include_enhanced=not args.no_enhanced, mirror_train=not args.no_mirror)
    train_loader = to_loader(*splits["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = to_loader(*splits["val"], batch_size=1024, shuffle=False)

    model = (MLP if args.model == "mlp" else CNN)(dropout=args.dropout).to(device)
    optimizer = make_optimizer(args, model.parameters())
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
                 if args.scheduler == "plateau" else None)
    loss_fn = nn.CrossEntropyLoss()
    print(f"{run_id}: {count_params(model):,} params, train {len(splits['train'][1]):,} / "
          f"val {len(splits['val'][1]):,} samples, device {device}")

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.CHECKPOINTS_DIR / f"{run_id}.pt"
    history_fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]

    best = {"val_loss": float("inf")}
    epochs_without_improvement = 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val = run_epoch(model, val_loader, loss_fn, device)
        if scheduler:
            scheduler.step(val["loss"])

        if not args.no_log:
            append_csv(config.HISTORY_DIR / f"{run_id}.csv", history_fields, {
                "epoch": epoch, "lr": lr, "train_loss": f"{train['loss']:.4f}", "train_acc": f"{train['acc']:.4f}",
                "val_loss": f"{val['loss']:.4f}", "val_acc": f"{val['acc']:.4f}", "val_f1": f"{val['f1']:.4f}",
            })
        print(f"  epoch {epoch:2d}  train loss {train['loss']:.4f} acc {train['acc']:.3f}  |  "
              f"val loss {val['loss']:.4f} acc {val['acc']:.3f} F1 {val['f1']:.3f}  "
              f"({time.perf_counter() - start:.0f} s total)")

        if val["loss"] < best["val_loss"]:
            best = {"epoch": epoch, "train_loss": train["loss"], "val_loss": val["loss"],
                    "val_acc": val["acc"], "val_f1": val["f1"]}
            if not args.no_log:
                torch.save({"model_state": model.state_dict(), "args": vars(args), "epoch": epoch}, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  early stop: no val loss improvement for {args.patience} epochs")
                break
    seconds = time.perf_counter() - start
    if args.no_log:
        print(f"Test run done in {seconds:.0f} s, nothing logged.")
        return

    append_csv(config.RUNS_CSV, RUN_FIELDS, {
        "run_id": run_id, "model": args.model, "optimizer": args.optimizer, "lr": args.lr,
        "momentum": args.momentum if args.optimizer == "sgd" else "", "batch_size": args.batch_size,
        "dropout": args.dropout, "scheduler": args.scheduler, "max_epochs": args.epochs, "patience": args.patience,
        "epochs_run": epoch, "best_epoch": best["epoch"], "train_loss": f"{best['train_loss']:.4f}",
        "val_loss": f"{best['val_loss']:.4f}", "val_acc": f"{best['val_acc']:.4f}", "val_f1": f"{best['val_f1']:.4f}",
        "params": count_params(model), "train_seconds": f"{seconds:.0f}", "split": args.split,
        "include_enhanced": not args.no_enhanced, "mirror_train": not args.no_mirror, "seed": args.seed,
        "notes": args.notes,
    })
    print(f"Best epoch {best['epoch']}: val acc {best['val_acc']:.3f}, val macro F1 {best['val_f1']:.3f} "
          f"({seconds:.0f} s). Logged to {config.RUNS_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
