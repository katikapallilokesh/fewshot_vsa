"""
prepare.py
----------
Merges FER2013 (train+test) and CK+ into one pool, then splits by emotion
into two disjoint datasets exactly as in the paper:

    Pretrain  (transfer learning)  : angry, disgust, fear, surprise
    Few-shot  (fine-tuning target) : happy, sad, neutral
    Unlabelled (pseudo-labeling)   : AffectNet — happy, sad, neutral only

Output folder layout:

    data/
        prepared/
            pretrain/<emotion>/*.{jpg,png}
            fewshot/<emotion>/*.{jpg,png}
        unlabelled/<emotion>/*.{jpg,png}   ← symlinked/copied from affectnet

Run:
    python -m src.data.prepare --data_root data
"""

import argparse
import random
import shutil
from pathlib import Path

PRETRAIN_EMOTIONS  = {"angry", "disgust", "fear", "surprise"}
FEWSHOT_EMOTIONS   = {"happy", "sad", "neutral"}
UNLABELLED_EMOTIONS = FEWSHOT_EMOTIONS   # AffectNet subset used for pseudo-labeling

MERGED_SOURCES = ["fer2013/train", "fer2013/test", "ck+"]


def merge_and_split(data_root: str = "data", seed: int = 42) -> None:
    random.seed(seed)
    data_root = Path(data_root)
    raw_dir   = data_root / "raw"
    out_dir   = data_root / "prepared"

    pretrain_dir  = out_dir / "pretrain"
    fewshot_dir   = out_dir / "fewshot"
    unlabelled_dir = data_root / "unlabelled"

    # -----------------------------------------------------------------------
    # 1. Merge FER2013 (train + test) + CK+ → pretrain / fewshot by emotion
    # -----------------------------------------------------------------------
    all_emotions = PRETRAIN_EMOTIONS | FEWSHOT_EMOTIONS

    counts: dict[str, int] = {}
    for emotion in all_emotions:
        # decide output folder
        if emotion in PRETRAIN_EMOTIONS:
            dest = pretrain_dir / emotion
        else:
            dest = fewshot_dir / emotion
        dest.mkdir(parents=True, exist_ok=True)

        n = 0
        for source in MERGED_SOURCES:
            src_emotion = raw_dir / source / emotion
            if not src_emotion.exists():
                continue
            for img_path in src_emotion.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                # prefix with source name to avoid filename collisions
                prefix = source.replace("/", "_")
                dest_name = f"{prefix}_{img_path.name}"
                shutil.copy(img_path, dest / dest_name)
                n += 1
        counts[emotion] = n

    print("\n[prepare] merged image counts:")
    for emotion, n in sorted(counts.items()):
        split = "pretrain" if emotion in PRETRAIN_EMOTIONS else "fewshot"
        print(f"  {emotion:<12} {split:<10} {n:>5} images")

    # -----------------------------------------------------------------------
    # 2. Copy AffectNet (happy, sad, neutral only) → unlabelled/
    # -----------------------------------------------------------------------
    aff_raw = raw_dir / "affectnet"
    for emotion in UNLABELLED_EMOTIONS:
        src = aff_raw / emotion
        dest = unlabelled_dir / emotion
        dest.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"[prepare] AffectNet/{emotion} not found, skipping.")
            continue
        n = 0
        for img_path in src.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            shutil.copy(img_path, dest / img_path.name)
            n += 1
        print(f"  {emotion:<12} unlabelled  {n:>5} images")

    print(f"\n[done] prepared data written to {out_dir}/")
    print(f"[done] unlabelled data written to {unlabelled_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    merge_and_split(args.data_root, args.seed)
