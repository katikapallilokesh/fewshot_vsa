"""
trainer.py
----------
Generic Trainer class used for both pretraining and few-shot fine-tuning.
Handles:
    - training loop with Adam + ReduceLROnPlateau
    - early stopping (optional)
    - best-model checkpointing
    - per-epoch metric logging
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, val_metric: float) -> bool:
        if self.best is None or val_metric > self.best + self.min_delta:
            self.best    = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class Trainer:
    """
    Args:
        model          : nn.Module (DCNN)
        train_loader   : DataLoader for training data
        val_loader     : DataLoader for validation data
        checkpoint_dir : directory to save best model weights
        run_name       : used as the checkpoint filename stem
        lr             : initial learning rate for Adam
        epochs         : maximum number of epochs
        patience       : early-stopping patience (None = disabled)
        device         : 'cuda' / 'cpu' / 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str | Path = "checkpoints",
        run_name: str = "model",
        lr: float = 1e-3,
        epochs: int = 100,
        patience: Optional[int] = 30,
        device: Optional[str] = None,
    ):
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name      = run_name
        self.epochs        = epochs
        self.device        = torch.device(
            device if device else
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-7, verbose=True
        )
        self.early_stopping = EarlyStopping(patience=patience) if patience else None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, list[float]]:
        best_val_acc = 0.0
        checkpoint_path = self.checkpoint_dir / f"{self.run_name}.pt"

        print(f"\n[trainer] device={self.device}  epochs={self.epochs}")
        print(f"          checkpoint → {checkpoint_path}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(train=True)
            val_loss,   val_acc   = self._run_epoch(train=False)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step(val_acc)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:>3}/{self.epochs} | "
                f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f}  acc {val_acc:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(checkpoint_path, val_acc)

            # Early stopping
            if self.early_stopping and self.early_stopping(val_acc):
                print(f"[trainer] early stopping at epoch {epoch}.")
                break

        # Restore best weights
        self._load_checkpoint(checkpoint_path)
        print(f"\n[trainer] best val accuracy: {best_val_acc:.4f}")
        return self.history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, train: bool) -> tuple[float, float]:
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader

        total_loss = 0.0
        correct    = 0
        total      = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds      = logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += images.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, path: Path, val_acc: float) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "num_classes":  self.model.num_classes,
            "val_acc":      val_acc,
        }, path)

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
