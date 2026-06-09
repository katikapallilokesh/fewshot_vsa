"""
evaluate.py
-----------
Evaluation utilities used after each training stage.

Functions
---------
evaluate_model      — accuracy, precision, recall, F1, ROC-AUC (aggregate)
evaluate_per_class  — same metrics broken down by class
confusion_matrix_fig — matplotlib confusion matrix figure
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def _predict(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_pred, y_proba).
    Works with datasets that return (image, int_label) tuples.
    """
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()

    all_true, all_pred, all_proba = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        all_proba.append(probs)
        all_pred.extend(preds.tolist())
        if isinstance(labels, torch.Tensor):
            all_true.extend(labels.numpy().tolist())
        else:
            all_true.extend(labels)

    return (
        np.array(all_true),
        np.array(all_pred),
        np.vstack(all_proba),
    )


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """
    Aggregate metrics over all classes.

    Returns dict with keys:
        accuracy, precision, recall, f1, roc_auc, loss (cross-entropy)
    """
    device = device or next(model.parameters()).device
    y_true, y_pred, y_proba = _predict(model, dataset, batch_size, device)
    num_classes = y_proba.shape[1]

    # Cross-entropy loss
    log_proba = np.log(np.clip(y_proba, 1e-9, 1.0))
    loss = -log_proba[np.arange(len(y_true)), y_true.astype(int)].mean()

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "loss":      float(loss),
    }

    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        metrics["roc_auc"] = float(
            roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def evaluate_per_class(
    model: nn.Module,
    dataset: Dataset,
    class_names: list[str],
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> dict[str, dict[str, float]]:
    """
    Per-class metrics.  Returns:
        { "happy": {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
          "sad":   {...},
          ... }
    """
    device = device or next(model.parameters()).device
    y_true, y_pred, _ = _predict(model, dataset, batch_size, device)

    results = {}
    for class_idx, class_name in enumerate(class_names):
        mask  = y_true == class_idx
        if mask.sum() == 0:
            results[class_name] = {"accuracy": 0., "precision": 0., "recall": 0., "f1": 0.}
            continue
        y_t = y_true[mask]
        y_p = y_pred[mask]
        results[class_name] = {
            "accuracy":  float((y_p == class_idx).mean()),
            "precision": float(precision_score(y_t, y_p, labels=[class_idx],
                                               average="micro", zero_division=0)),
            "recall":    float(recall_score(y_t, y_p, labels=[class_idx],
                                            average="micro", zero_division=0)),
            "f1":        float(f1_score(y_t, y_p, labels=[class_idx],
                                        average="micro", zero_division=0)),
        }

    return results


def confusion_matrix_fig(
    model: nn.Module,
    dataset: Dataset,
    class_names: list[str],
    title: str = "Confusion Matrix",
    device: Optional[torch.device] = None,
) -> plt.Figure:
    """Returns a matplotlib Figure of the confusion matrix."""
    device = device or next(model.parameters()).device
    y_true, y_pred, _ = _predict(model, dataset, device=device)

    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{title}\n")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig
