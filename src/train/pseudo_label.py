"""
pseudo_label.py
---------------
Implements the pseudo-labeling strategy from the paper:

    1. Run the current model over the unlabelled AffectNet pool.
    2. Keep only predictions with softmax confidence ≥ threshold (default 0.99).
    3. Select at most `floor(k_shot * multiplier)` samples per class so the
       added pseudo-data stays proportional to the labeled pool.
    4. Remove selected samples from the unlabelled pool (no re-use across
       single-pass → iterative rounds).

Used twice per k-shot iteration:
    - Single SS   : model = baseline fine-tuned,  multiplier=0.25, threshold=0.99
    - Iterative SS: model = single-SS fine-tuned, multiplier=0.30, threshold=0.99
"""

import math
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from src.data.datasets import UnlabelledDataset, get_eval_transform


@torch.no_grad()
def pseudo_label(
    model: nn.Module,
    unlabelled_ds: UnlabelledDataset,
    class_to_idx: dict[str, int],
    k_shot: int,
    multiplier: float = 0.25,
    threshold: float = 0.99,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> tuple[Dataset, UnlabelledDataset, dict]:
    """
    Parameters
    ----------
    model           : fine-tuned DCNN (already on device)
    unlabelled_ds   : UnlabelledDataset — pool to draw from
    class_to_idx    : mapping from emotion string → int label (must match model head)
    k_shot          : number of labeled samples per class used in this iteration
    multiplier      : pseudo samples = min(min_class_count, ceil(k_shot * multiplier))
    threshold       : minimum softmax confidence to accept a pseudo-label
    batch_size      : inference batch size
    device          : torch.device (auto-detected if None)

    Returns
    -------
    pseudo_dataset      : Subset of unlabelled_ds with pseudo-labels injected
    remaining_pool      : UnlabelledDataset with selected samples removed
    stats               : dict with sample counts and pseudo-label accuracy
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    loader = DataLoader(unlabelled_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # -----------------------------------------------------------------------
    # 1. Forward pass — collect (index, predicted_class, confidence, true_emotion)
    # -----------------------------------------------------------------------
    all_probs:        list[torch.Tensor] = []
    all_true_emotions: list[str]          = []

    for images, true_emotions in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_true_emotions.extend(true_emotions)

    all_probs_t = torch.cat(all_probs, dim=0)           # (N, num_classes)
    confidences, pred_labels = all_probs_t.max(dim=1)   # (N,), (N,)

    # -----------------------------------------------------------------------
    # 2. Filter by confidence threshold
    # -----------------------------------------------------------------------
    high_conf_mask = confidences >= threshold
    high_conf_indices    = high_conf_mask.nonzero(as_tuple=True)[0].tolist()
    high_conf_pred_names = [idx_to_class[pred_labels[i].item()] for i in high_conf_indices]
    high_conf_true_names = [all_true_emotions[i] for i in high_conf_indices]

    # Group indices by predicted class
    class_candidates: dict[str, list[int]] = defaultdict(list)
    for global_idx, pred_name in zip(high_conf_indices, high_conf_pred_names):
        class_candidates[pred_name].append(global_idx)

    # -----------------------------------------------------------------------
    # 3. Determine how many samples to take per class
    # -----------------------------------------------------------------------
    class_counts = [len(v) for v in class_candidates.values()]
    if not class_counts:
        print("[pseudo_label] no high-confidence samples found — skipping.")
        return _empty_pseudo_dataset(), unlabelled_ds, {"selected": 0}

    min_count  = min(class_counts)
    num_samples = min(min_count, math.ceil(k_shot * multiplier))
    num_samples = max(num_samples, 1)

    # -----------------------------------------------------------------------
    # 4. Select samples and build pseudo-labelled dataset
    # -----------------------------------------------------------------------
    selected_global_indices: list[int] = []
    selected_pseudo_labels:  list[int] = []

    for class_name, candidates in class_candidates.items():
        chosen = candidates[:num_samples]   # already ordered by iteration
        selected_global_indices.extend(chosen)
        label_int = class_to_idx[class_name]
        selected_pseudo_labels.extend([label_int] * len(chosen))

    # Accuracy of pseudo-labels against true AffectNet labels
    selected_true = [all_true_emotions[i] for i in selected_global_indices]
    selected_pred = [idx_to_class[class_to_idx[all_true_emotions[i]]]   # map true→int→str
                     if all_true_emotions[i] in class_to_idx else "?"
                     for i in selected_global_indices]
    # actual pseudo accuracy: compare predicted name vs true name
    selected_pred_names = [idx_to_class[selected_pseudo_labels[j]]
                           for j in range(len(selected_global_indices))]
    correct = sum(p == t for p, t in zip(selected_pred_names, selected_true))
    pseudo_accuracy = correct / len(selected_global_indices) * 100 if selected_global_indices else 0.0

    # Wrap as a dataset with hard pseudo-labels
    pseudo_ds = _PseudoLabelDataset(
        unlabelled_ds, selected_global_indices, selected_pseudo_labels
    )

    # -----------------------------------------------------------------------
    # 5. Remove selected from the unlabelled pool
    # -----------------------------------------------------------------------
    selected_set = set(selected_global_indices)
    remaining_pool = unlabelled_ds.remove_indices(selected_set)

    stats = {
        "selected_per_class": num_samples,
        "total_selected":     len(selected_global_indices),
        "pseudo_accuracy":    round(pseudo_accuracy, 2),
        "remaining_pool":     len(remaining_pool),
    }
    print(
        f"[pseudo_label] selected {len(selected_global_indices)} samples "
        f"({num_samples}/class) | pseudo-accuracy={pseudo_accuracy:.1f}% | "
        f"pool remaining={len(remaining_pool)}"
    )
    return pseudo_ds, remaining_pool, stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _PseudoLabelDataset(Dataset):
    """
    A view over an UnlabelledDataset that returns (image, int_label) tuples
    using the pseudo-labels assigned by the model.
    """

    def __init__(
        self,
        base: UnlabelledDataset,
        indices: list[int],
        pseudo_labels: list[int],
    ):
        assert len(indices) == len(pseudo_labels)
        self.base          = base
        self.indices       = indices
        self.pseudo_labels = pseudo_labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, _ = self.base[self.indices[idx]]   # ignore true emotion string
        return image, self.pseudo_labels[idx]


def _empty_pseudo_dataset() -> Dataset:
    class _Empty(Dataset):
        def __len__(self): return 0
        def __getitem__(self, i): raise IndexError
    return _Empty()


def combine_labelled_and_pseudo(
    labelled_dataset: Dataset,
    pseudo_dataset: Dataset,
) -> Dataset:
    """Concatenate a labelled k-shot subset with a pseudo-labelled dataset."""
    if len(pseudo_dataset) == 0:
        return labelled_dataset
    return ConcatDataset([labelled_dataset, pseudo_dataset])
