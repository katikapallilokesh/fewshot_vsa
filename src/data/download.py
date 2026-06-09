"""
download.py
-----------
Downloads FER2013, CK+, and AffectNet from Kaggle using the Kaggle API,
then organises them into a unified folder structure:

    data/raw/
        fer2013/train/<emotion>/*.jpg
        fer2013/test/<emotion>/*.jpg
        ck+/<emotion>/*.jpg
        affectnet/<emotion>/*.jpg

Run once before anything else:
    python -m src.data.download --kaggle_json path/to/kaggle.json
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

KAGGLE_DATASETS = {
    "fer2013":   "msambare/fer2013",
    "ck+":       "davilsena/ckdataset",
    "affectnet": "noamsegal/affectnet-training-data",
}

# Emotions kept across the project (contempt excluded from AffectNet)
VALID_EMOTIONS = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}

# CK+ numeric → string mapping (label 7 = contempt, dropped)
CK_LABEL_MAP = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
                4: "sad",   5: "surprise", 6: "neutral"}


def _setup_kaggle_credentials(kaggle_json: str) -> None:
    """Copy kaggle.json to ~/.kaggle/ so the API can authenticate."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    dest = kaggle_dir / "kaggle.json"
    shutil.copy(kaggle_json, dest)
    dest.chmod(0o600)
    print(f"[kaggle] credentials written to {dest}")


def _download_dataset(slug: str, dest_dir: Path) -> Path:
    """Download and unzip a Kaggle dataset, return the unzipped folder."""
    import kaggle  # imported late so credential setup happens first
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] downloading {slug} → {dest_dir}")
    kaggle.api.dataset_download_files(slug, path=str(dest_dir), unzip=True, quiet=False)
    return dest_dir


# ---------------------------------------------------------------------------
# Per-dataset organisation helpers
# ---------------------------------------------------------------------------

def _organise_fer2013(raw_dir: Path, out_dir: Path) -> None:
    """
    FER2013 arrives as:   raw_dir/train/<Emotion>/*.jpg
                          raw_dir/test/<Emotion>/*.jpg
    We lowercase the emotion folder names and drop any not in VALID_EMOTIONS.
    """
    for split in ("train", "test"):
        src_split = raw_dir / split
        if not src_split.exists():
            print(f"[fer2013] {src_split} not found, skipping {split}")
            continue
        for emotion_dir in src_split.iterdir():
            emotion = emotion_dir.name.lower()
            if emotion not in VALID_EMOTIONS:
                continue
            out_emotion = out_dir / split / emotion
            out_emotion.mkdir(parents=True, exist_ok=True)
            for img in emotion_dir.glob("*"):
                shutil.copy(img, out_emotion / img.name)
    print("[fer2013] organised.")


def _organise_ckplus(raw_dir: Path, out_dir: Path) -> None:
    """
    CK+ dataset arrives as a single CSV: ckextended.csv
    Columns: emotion (int), pixels (space-separated), Usage
    We decode pixels → PNG and save under out_dir/<emotion>/.
    """
    import numpy as np
    import pandas as pd
    from PIL import Image

    csv_path = next(raw_dir.glob("**/*.csv"), None)
    if csv_path is None:
        print("[ck+] CSV not found, skipping.")
        return

    df = pd.read_csv(csv_path)
    # Drop contempt (label 7)
    df = df[df["emotion"] != 7].copy()
    df["emotion"] = df["emotion"].map(CK_LABEL_MAP)

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in df.iterrows():
        emotion = row["emotion"]
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        out_emotion = out_dir / emotion
        out_emotion.mkdir(exist_ok=True)
        img = Image.fromarray(pixels, mode="L")
        img.save(out_emotion / f"ck_{idx:05d}.png")

    print(f"[ck+] organised {len(df)} images.")


def _organise_affectnet(raw_dir: Path, out_dir: Path) -> None:
    """
    AffectNet arrives as subfolders per emotion.
    We lowercase, rename 'anger' → 'angry', and drop contempt.
    """
    rename = {"anger": "angry"}
    out_dir.mkdir(parents=True, exist_ok=True)

    for emotion_dir in raw_dir.iterdir():
        if not emotion_dir.is_dir():
            continue
        emotion = rename.get(emotion_dir.name.lower(), emotion_dir.name.lower())
        if emotion not in VALID_EMOTIONS:
            continue
        out_emotion = out_dir / emotion
        out_emotion.mkdir(exist_ok=True)
        for img in emotion_dir.glob("*"):
            shutil.copy(img, out_emotion / img.name)

    print("[affectnet] organised.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_and_organise(kaggle_json: str, data_root: str = "data") -> None:
    data_root = Path(data_root)
    download_dir = data_root / "downloads"
    raw_dir      = data_root / "raw"

    _setup_kaggle_credentials(kaggle_json)

    # FER2013
    fer_dl  = _download_dataset(KAGGLE_DATASETS["fer2013"],   download_dir / "fer2013")
    _organise_fer2013(fer_dl, raw_dir / "fer2013")

    # CK+
    ck_dl   = _download_dataset(KAGGLE_DATASETS["ck+"],       download_dir / "ck+")
    _organise_ckplus(ck_dl, raw_dir / "ck+")

    # AffectNet
    aff_dl  = _download_dataset(KAGGLE_DATASETS["affectnet"], download_dir / "affectnet")
    _organise_affectnet(aff_dl, raw_dir / "affectnet")

    print(f"\n[done] raw data ready under {raw_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle_json", required=True,
                        help="Path to your kaggle.json credentials file")
    parser.add_argument("--data_root", default="data",
                        help="Root folder where data will be stored (default: data/)")
    args = parser.parse_args()
    download_and_organise(args.kaggle_json, args.data_root)
