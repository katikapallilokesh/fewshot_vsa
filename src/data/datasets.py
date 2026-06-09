"""
datasets.py
-----------
PyTorch Dataset classes and DataLoader factories for all three data roles:

    FERDataset          — generic image-folder dataset with optional transform
    KShotSampler        — samples exactly k images per class from a dataset
    FERDataLoader       — factory returning (train_loader, val_loader)
    UnlabelledDataset   — AffectNet images without labels (for pseudo-labeling)
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15),
                                shear=0.15, scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_eval_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class FERDataset(Dataset):
    """
    Reads images from a folder structured as:
        root/<emotion>/image.jpg

    Args:
        root        : path to the split folder (e.g. data/prepared/pretrain)
        transform   : torchvision transform applied to each image
        emotions    : if given, only these emotion classes are loaded
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        emotions: Optional[list[str]] = None,
    ):
        self.root = Path(root)
        self.transform = transform or get_eval_transform()

        emotion_dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and (emotions is None or d.name in emotions)
        )
        if not emotion_dirs:
            raise ValueError(f"No emotion subdirectories found under {self.root}")

        self.classes   = [d.name for d in emotion_dirs]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for emotion_dir in emotion_dirs:
            label = self.class_to_idx[emotion_dir.name]
            for img_path in sorted(emotion_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")   # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label

    # convenience: indices grouped by class
    def indices_by_class(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(self.samples):
            groups[label].append(idx)
        return dict(groups)


# ---------------------------------------------------------------------------
# K-Shot Sampler
# ---------------------------------------------------------------------------

class KShotSampler:
    """
    Given a FERDataset, selects exactly `k` samples per class (or fewer
    if the class has fewer than k samples).  Returns a Subset.
    """

    def __init__(self, dataset: FERDataset, k: int, seed: int = 42):
        self.dataset = dataset
        self.k = k
        self.seed = seed

    def sample(self) -> Subset:
        rng = random.Random(self.seed)
        groups = self.dataset.indices_by_class()
        selected: list[int] = []
        for label in sorted(groups):
            pool = groups[label]
            n = min(self.k, len(pool))
            selected.extend(rng.sample(pool, n))
        return Subset(self.dataset, selected)

    def sample_and_remove(self) -> tuple[Subset, "KShotSampler"]:
        """
        Returns a k-shot Subset and a new KShotSampler over the remaining
        indices (used to draw validation / test sets without overlap).
        """
        rng = random.Random(self.seed)
        groups = self.dataset.indices_by_class()
        selected: list[int] = []
        remaining: list[int] = []
        for label in sorted(groups):
            pool = groups[label]
            n = min(self.k, len(pool))
            chosen = rng.sample(pool, n)
            selected.extend(chosen)
            remaining.extend(i for i in pool if i not in set(chosen))

        subset = Subset(self.dataset, selected)
        remainder_dataset = Subset(self.dataset, remaining)
        return subset, remainder_dataset


# ---------------------------------------------------------------------------
# Unlabelled Dataset (AffectNet for pseudo-labeling)
# ---------------------------------------------------------------------------

class UnlabelledDataset(Dataset):
    """
    Loads images from data/unlabelled/<emotion>/ — the 'emotion' folder name
    is stored as metadata for post-hoc accuracy evaluation of pseudo-labels,
    but is NOT used as a training signal.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        emotions: Optional[list[str]] = None,
    ):
        self.root = Path(root)
        self.transform = transform or get_eval_transform()

        self.samples: list[tuple[Path, str]] = []
        for emotion_dir in sorted(self.root.iterdir()):
            if not emotion_dir.is_dir():
                continue
            if emotions and emotion_dir.name not in emotions:
                continue
            for img_path in sorted(emotion_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, emotion_dir.name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img_path, emotion = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, emotion

    def remove_indices(self, indices: set[int]) -> "UnlabelledDataset":
        """Return a new UnlabelledDataset with specified indices removed."""
        new_ds = UnlabelledDataset.__new__(UnlabelledDataset)
        new_ds.root = self.root
        new_ds.transform = self.transform
        new_ds.samples = [s for i, s in enumerate(self.samples) if i not in indices]
        return new_ds


# ---------------------------------------------------------------------------
# DataLoader Factories
# ---------------------------------------------------------------------------

def make_pretrain_loaders(
    data_root: str | Path = "data",
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for the pretrain split
    (angry, disgust, fear, surprise).
    """
    from torch.utils.data import random_split

    data_root = Path(data_root)
    full_ds = FERDataset(
        root=data_root / "prepared" / "pretrain",
        transform=None,   # train transform applied below after split
    )

    n_val  = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    # Apply different transforms to each split
    train_ds.dataset = _TransformWrapper(full_ds, get_train_transform())
    # val uses eval transform — re-wrap just the val indices
    val_base = FERDataset(
        root=data_root / "prepared" / "pretrain",
        transform=get_eval_transform(),
    )
    val_ds = Subset(val_base, val_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[dataloader] pretrain — train: {len(train_ds)}, val: {len(val_ds)}")
    print(f"             classes: {full_ds.classes}")
    return train_loader, val_loader


def make_fewshot_loaders(
    data_root: str | Path = "data",
    k_shot: int = 10,
    val_size: int = 15,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, Dataset]:
    """
    Returns (train_loader, val_loader, test_dataset) for few-shot fine-tuning.

    - train_loader : k samples per class (with augmentation)
    - val_loader   : val_size samples per class (no overlap with train)
    - test_dataset : remaining samples (used after training for final eval)
    """
    data_root = Path(data_root)

    # Base dataset with eval transform — train transform applied via wrapper
    base_ds = FERDataset(
        root=data_root / "prepared" / "fewshot",
        transform=get_eval_transform(),
    )

    # k-shot train split
    train_sampler = KShotSampler(base_ds, k=k_shot, seed=seed)
    train_subset, remaining = train_sampler.sample_and_remove()

    # Wrap train indices with augmentation transform
    aug_ds = FERDataset(
        root=data_root / "prepared" / "fewshot",
        transform=get_train_transform(),
    )
    train_aug = Subset(aug_ds, train_subset.indices)

    # val split from remaining
    remaining_ds = FERDataset(
        root=data_root / "prepared" / "fewshot",
        transform=get_eval_transform(),
    )
    # collect remaining indices
    train_idx_set = set(train_subset.indices)
    remaining_indices = [i for i in range(len(base_ds)) if i not in train_idx_set]
    remaining_ds_subset = Subset(remaining_ds, remaining_indices)

    val_sampler = KShotSampler(
        _IndexedSubsetDataset(remaining_ds, remaining_indices),
        k=val_size, seed=seed + 1
    )
    val_subset = val_sampler.sample()
    val_indices_local = val_subset.indices
    val_global_indices = [remaining_indices[i] for i in val_indices_local]

    val_ds   = Subset(remaining_ds, val_global_indices)
    val_idx_set = set(val_global_indices)
    test_indices = [i for i in remaining_indices if i not in val_idx_set]
    test_ds  = Subset(remaining_ds, test_indices)

    train_loader = DataLoader(train_aug, batch_size=min(batch_size, len(train_aug)),
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[dataloader] few-shot k={k_shot} — train: {len(train_aug)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_ds


def make_unlabelled_loader(
    data_root: str | Path = "data",
    batch_size: int = 64,
    num_workers: int = 4,
    emotions: Optional[list[str]] = None,
) -> DataLoader:
    data_root = Path(data_root)
    ds = UnlabelledDataset(
        root=data_root / "unlabelled",
        transform=get_eval_transform(),
        emotions=emotions,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _TransformWrapper(Dataset):
    """Wraps a FERDataset with a different transform (avoids duplicating disk reads)."""
    def __init__(self, dataset: FERDataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        img = Image.open(img_path).convert("L")
        return self.transform(img), label


class _IndexedSubsetDataset(Dataset):
    """A FERDataset view restricted to a given list of global indices."""
    def __init__(self, dataset: FERDataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.samples = [dataset.samples[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def indices_by_class(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for local_idx, global_idx in enumerate(self.indices):
            _, label = self.dataset.samples[global_idx]
            groups[label].append(local_idx)
        return dict(groups)
