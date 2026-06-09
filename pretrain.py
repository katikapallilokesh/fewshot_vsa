"""
pretrain.py
-----------
Stage 1: Train the DCNN backbone on the pretrain split
(angry, disgust, fear, surprise) and save the best checkpoint.

Usage:
    python pretrain.py
    python pretrain.py --data_root /path/to/data --epochs 100 --batch_size 32
"""

import argparse
import json
from pathlib import Path

import torch

from src.data.datasets import make_pretrain_loaders
from src.models.dcnn import build_pretrain_model
from src.train.trainer import Trainer
from src.utils.evaluate import evaluate_model, confusion_matrix_fig
from src.utils.plot import plot_training_history


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain DCNN backbone")
    p.add_argument("--data_root",      default="data")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--plot_dir",       default="plots/pretrain")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--val_split",      type=float, default=0.1)
    p.add_argument("--patience",       type=int,   default=30)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader = make_pretrain_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Infer number of classes from the dataset
    num_classes  = len(train_loader.dataset.dataset.classes)
    class_names  = train_loader.dataset.dataset.classes
    print(f"[pretrain] classes ({num_classes}): {class_names}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_pretrain_model(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
        run_name="backbone",
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
    )
    history = trainer.fit()

    # ------------------------------------------------------------------
    # Evaluate on validation set
    # ------------------------------------------------------------------
    device = trainer.device
    val_ds = val_loader.dataset

    metrics = evaluate_model(model, val_ds, device=device)
    print("\n[pretrain] Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")

    # Save metrics
    metrics_path = Path(args.checkpoint_dir) / "pretrain_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"val": metrics, "classes": class_names}, f, indent=2)

    # Confusion matrix
    cm_fig = confusion_matrix_fig(model, val_ds, class_names,
                                   title="Pretrain — Validation", device=device)
    cm_fig.savefig(plot_dir / "confusion_matrix.png", dpi=150)

    # Training curves
    plot_training_history(history, title_prefix="Pretrain",
                          save_path=plot_dir / "training_curves.png")

    print(f"\n[pretrain] done. Checkpoint → {args.checkpoint_dir}/backbone.pt")


if __name__ == "__main__":
    main()
