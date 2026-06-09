"""
run_experiment.py
-----------------
Stage 2: Few-shot experiments.

For each k in k_shots, runs three conditions in sequence:
    1. Baseline     — fine-tune on k labeled samples per class
    2. Single SS    — baseline + one round of pseudo-labeling from AffectNet
    3. Iterative SS — single SS + second round of pseudo-labeling

Results are saved per-k to results/<k>/ and aggregated to results/summary.json.
Plots are saved to plots/fewshot/.

Usage:
    python run_experiment.py
    python run_experiment.py --backbone checkpoints/backbone.pt --k_shots 1 5 10 20
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.datasets import (
    FERDataset,
    UnlabelledDataset,
    get_eval_transform,
    get_train_transform,
    make_fewshot_loaders,
    make_unlabelled_loader,
)
from src.models.dcnn import build_fewshot_model
from src.train.pseudo_label import combine_labelled_and_pseudo, pseudo_label
from src.train.trainer import Trainer
from src.utils.evaluate import confusion_matrix_fig, evaluate_model, evaluate_per_class
from src.utils.plot import plot_all_results, plot_training_history


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Few-shot FER experiments")
    p.add_argument("--backbone",        default="checkpoints/backbone.pt")
    p.add_argument("--data_root",       default="data")
    p.add_argument("--results_dir",     default="results")
    p.add_argument("--checkpoint_dir",  default="checkpoints/fewshot")
    p.add_argument("--plot_dir",        default="plots/fewshot")
    p.add_argument("--k_shots", nargs="+", type=int,
                   default=[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    p.add_argument("--val_size",         type=int,   default=15)
    p.add_argument("--test_size",        type=int,   default=15)
    p.add_argument("--epochs",           type=int,   default=20)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--threshold",        type=float, default=0.99)
    p.add_argument("--single_mult",      type=float, default=0.25,
                   help="Pseudo-label multiplier for single SS round")
    p.add_argument("--iterative_mult",   type=float, default=0.30,
                   help="Pseudo-label multiplier for iterative SS round")
    p.add_argument("--num_workers",      type=int,   default=2)
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: build a DataLoader from an arbitrary Dataset
# ---------------------------------------------------------------------------

def _to_loader(dataset, batch_size, shuffle=True, num_workers=2):
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)),
                      shuffle=shuffle, num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Single k-shot run
# ---------------------------------------------------------------------------

def run_one_k(
    k: int,
    args,
    fewshot_ds_full: FERDataset,
    unlabelled_ds: UnlabelledDataset,
    class_names: list[str],
    class_to_idx: dict[str, int],
    device: torch.device,
    plot_dir: Path,
    ckpt_dir: Path,
    results_dir: Path,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  k = {k}")
    print(f"{'='*60}")

    from src.data.datasets import KShotSampler, _IndexedSubsetDataset
    from torch.utils.data import Subset

    # ----------------------------------------------------------------
    # Sample train / val / test splits (no overlap)
    # ----------------------------------------------------------------
    rng_seed = args.seed

    # Train: k per class
    train_sampler = KShotSampler(fewshot_ds_full, k=k, seed=rng_seed)
    train_subset_idx  = train_sampler.sample().indices
    train_idx_set     = set(train_subset_idx)

    # Val: val_size per class from remaining
    remaining_indices = [i for i in range(len(fewshot_ds_full)) if i not in train_idx_set]
    rem_ds = _IndexedSubsetDataset(fewshot_ds_full, remaining_indices)
    val_sampler  = KShotSampler(rem_ds, k=args.val_size, seed=rng_seed + 1)
    val_local_idx = val_sampler.sample().indices
    val_global_idx = [remaining_indices[i] for i in val_local_idx]
    val_idx_set    = set(val_global_idx)

    # Test: test_size per class from what's left
    after_val   = [i for i in remaining_indices if i not in val_idx_set]
    after_val_ds = _IndexedSubsetDataset(fewshot_ds_full, after_val)
    test_sampler = KShotSampler(after_val_ds, k=args.test_size, seed=rng_seed + 2)
    test_local_idx  = test_sampler.sample().indices
    test_global_idx = [after_val[i] for i in test_local_idx]

    # Datasets with correct transforms
    aug_ds   = FERDataset(fewshot_ds_full.root, transform=get_train_transform())
    eval_ds  = FERDataset(fewshot_ds_full.root, transform=get_eval_transform())

    train_ds = Subset(aug_ds,  train_subset_idx)
    val_ds   = Subset(eval_ds, val_global_idx)
    test_ds  = Subset(eval_ds, test_global_idx)

    train_loader = _to_loader(train_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = _to_loader(val_ds,   args.batch_size, shuffle=False, num_workers=args.num_workers)

    k_results = {}

    # ----------------------------------------------------------------
    # 1. BASELINE
    # ----------------------------------------------------------------
    print(f"\n--- Baseline ---")
    baseline_model = build_fewshot_model(
        args.backbone, num_classes=len(class_names), num_unfrozen_blocks=1
    )
    ckpt_baseline = ckpt_dir / f"baseline_k{k}.pt"
    trainer = Trainer(baseline_model, train_loader, val_loader,
                      checkpoint_dir=ckpt_dir, run_name=f"baseline_k{k}",
                      lr=args.lr, epochs=args.epochs, patience=None, device=str(device))
    history_b = trainer.fit()

    plot_training_history(history_b, f"Baseline {k}-shot",
                          save_path=plot_dir / f"baseline_k{k}_curves.png")
    cm_b = confusion_matrix_fig(baseline_model, test_ds, class_names,
                                 title=f"Baseline {k}-shot", device=device)
    cm_b.savefig(plot_dir / f"baseline_k{k}_cm.png")

    k_results["baseline"] = {
        "train": evaluate_model(baseline_model, train_ds, device=device),
        "test":  evaluate_model(baseline_model, test_ds,  device=device),
        "per_class_test": evaluate_per_class(baseline_model, test_ds, class_names, device=device),
    }

    # ----------------------------------------------------------------
    # 2. SINGLE SEMI-SUPERVISED
    # ----------------------------------------------------------------
    print(f"\n--- Single SS ---")
    # Fresh unlabelled pool for this k
    unlabelled_pool = UnlabelledDataset(
        root=Path(args.data_root) / "unlabelled",
        transform=get_eval_transform(),
        emotions=class_names,
    )

    pseudo_ds_single, unlabelled_pool_after_single, stats_single = pseudo_label(
        model=baseline_model,
        unlabelled_ds=unlabelled_pool,
        class_to_idx=class_to_idx,
        k_shot=k,
        multiplier=args.single_mult,
        threshold=args.threshold,
        device=device,
    )

    single_combined = combine_labelled_and_pseudo(
        Subset(aug_ds, train_subset_idx), pseudo_ds_single
    )
    single_loader = _to_loader(single_combined, args.batch_size, num_workers=args.num_workers)

    single_model = build_fewshot_model(
        str(ckpt_baseline), num_classes=len(class_names), num_unfrozen_blocks=1
    )
    trainer = Trainer(single_model, single_loader, val_loader,
                      checkpoint_dir=ckpt_dir, run_name=f"single_ss_k{k}",
                      lr=args.lr, epochs=args.epochs, patience=None, device=str(device))
    history_s = trainer.fit()

    plot_training_history(history_s, f"Single SS {k}-shot",
                          save_path=plot_dir / f"single_ss_k{k}_curves.png")
    cm_s = confusion_matrix_fig(single_model, test_ds, class_names,
                                 title=f"Single SS {k}-shot", device=device)
    cm_s.savefig(plot_dir / f"single_ss_k{k}_cm.png")

    k_results["single_ss"] = {
        "train":          evaluate_model(single_model, single_combined, device=device),
        "test":           evaluate_model(single_model, test_ds, device=device),
        "per_class_test": evaluate_per_class(single_model, test_ds, class_names, device=device),
        "pseudo_stats":   stats_single,
    }

    # ----------------------------------------------------------------
    # 3. ITERATIVE SEMI-SUPERVISED
    # ----------------------------------------------------------------
    print(f"\n--- Iterative SS ---")
    ckpt_single = ckpt_dir / f"single_ss_k{k}.pt"

    pseudo_ds_iter, _, stats_iter = pseudo_label(
        model=single_model,
        unlabelled_ds=unlabelled_pool_after_single,
        class_to_idx=class_to_idx,
        k_shot=k,
        multiplier=args.iterative_mult,
        threshold=args.threshold,
        device=device,
    )

    iter_combined = combine_labelled_and_pseudo(single_combined, pseudo_ds_iter)
    iter_loader   = _to_loader(iter_combined, args.batch_size, num_workers=args.num_workers)

    iter_model = build_fewshot_model(
        str(ckpt_single), num_classes=len(class_names), num_unfrozen_blocks=1
    )
    trainer = Trainer(iter_model, iter_loader, val_loader,
                      checkpoint_dir=ckpt_dir, run_name=f"iterative_ss_k{k}",
                      lr=args.lr, epochs=args.epochs, patience=None, device=str(device))
    history_i = trainer.fit()

    plot_training_history(history_i, f"Iterative SS {k}-shot",
                          save_path=plot_dir / f"iterative_ss_k{k}_curves.png")
    cm_i = confusion_matrix_fig(iter_model, test_ds, class_names,
                                 title=f"Iterative SS {k}-shot", device=device)
    cm_i.savefig(plot_dir / f"iterative_ss_k{k}_cm.png")

    k_results["iterative_ss"] = {
        "train":          evaluate_model(iter_model, iter_combined, device=device),
        "test":           evaluate_model(iter_model, test_ds,       device=device),
        "per_class_test": evaluate_per_class(iter_model, test_ds, class_names, device=device),
        "pseudo_stats":   stats_iter,
    }

    # Save per-k results JSON
    k_res_path = results_dir / f"k{k}.json"
    with open(k_res_path, "w") as f:
        json.dump(k_results, f, indent=2)

    print(f"\n[k={k}] test accuracy — "
          f"baseline={k_results['baseline']['test']['accuracy']:.4f}  "
          f"single={k_results['single_ss']['test']['accuracy']:.4f}  "
          f"iterative={k_results['iterative_ss']['test']['accuracy']:.4f}")

    return k_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    results_dir = Path(args.results_dir)
    plot_dir    = Path(args.plot_dir)
    ckpt_dir    = Path(args.checkpoint_dir)
    for d in (results_dir, plot_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[experiment] device={device}")

    # Build base dataset (eval transform — augmentation applied per-subset inside the loop)
    fewshot_ds_full = FERDataset(
        root=Path(args.data_root) / "prepared" / "fewshot",
        transform=get_eval_transform(),
    )
    class_names  = fewshot_ds_full.classes
    class_to_idx = fewshot_ds_full.class_to_idx
    print(f"[experiment] few-shot classes: {class_names}")

    # Run per k
    all_results: dict[int, dict] = {}
    for k in args.k_shots:
        all_results[k] = run_one_k(
            k=k, args=args,
            fewshot_ds_full=fewshot_ds_full,
            unlabelled_ds=None,   # built fresh inside run_one_k
            class_names=class_names,
            class_to_idx=class_to_idx,
            device=device,
            plot_dir=plot_dir,
            ckpt_dir=ckpt_dir,
            results_dir=results_dir,
        )

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    summary = {
        "test_accuracy":  {"baseline": [], "single": [], "iterative": []},
        "test_f1":        {"baseline": [], "single": [], "iterative": []},
        "test_loss":      {"baseline": [], "single": [], "iterative": []},
        "train_accuracy": {"baseline": [], "single": [], "iterative": []},
        "train_loss":     {"baseline": [], "single": [], "iterative": []},
        "pseudo_samples": {"single": [], "iterative": []},
    }

    for k in args.k_shots:
        r = all_results[k]
        for cond, key in [("baseline", "baseline"), ("single_ss", "single"), ("iterative_ss", "iterative")]:
            summary["test_accuracy"][key].append(r[cond]["test"]["accuracy"])
            summary["test_f1"][key].append(r[cond]["test"]["f1"])
            summary["test_loss"][key].append(r[cond]["test"]["loss"])
            summary["train_accuracy"][key].append(r[cond]["train"]["accuracy"])
            summary["train_loss"][key].append(r[cond]["train"]["loss"])

        summary["pseudo_samples"]["single"].append(
            r["single_ss"]["pseudo_stats"].get("total_selected", 0))
        summary["pseudo_samples"]["iterative"].append(
            r["iterative_ss"]["pseudo_stats"].get("total_selected", 0))

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Final comparison plots
    plot_all_results(summary, args.k_shots, plot_dir=plot_dir)
    print(f"\n[experiment] complete. Results → {results_dir}/  Plots → {plot_dir}/")


if __name__ == "__main__":
    main()
