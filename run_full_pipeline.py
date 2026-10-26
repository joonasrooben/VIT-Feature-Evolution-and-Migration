"""Command-line orchestration of the existing ImageNet-1K pipeline functions.

See FULL_PIPELINE_SETUP.md for the small activation-exporter edits to apply first.
This module does not read or execute a notebook.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import evo_aggregate_patterns as aggregate
import generate_acts_vit_fun_tiny as activations
import sae_track_method as tracking
from src.sae_utils import train_sae_fast


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["check", "export", "sae", "evaluate", "plot", "aggregate"])
    parser.add_argument("--config", type=Path, default=Path("sae_exp_config_base.yml"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--train-imglist", required=True, type=Path)
    parser.add_argument("--test-imglist", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=300,
                        help="Trainer epoch count; checkpoint IDs are 0..epochs-1.")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epoch-stride", type=int, default=1)
    parser.add_argument("--reference-epochs", default="",
                        help="Optional comma-separated IDs; default is the paper's dense/sparse schedule.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--reuse-activations", action="store_true",
                        help="Check existing exports instead of requiring exporter source edits.")
    return parser.parse_args()


def experiment(args):
    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if args.epochs < 2 or args.epoch_stride < 1:
        raise ValueError("Use epochs >= 2 and epoch-stride >= 1.")
    if cfg.get("sae_runs", 1) != 1:
        raise ValueError("Combined .npz filenames lack a run ID; run each SAE seed in a separate directory.")
    epochs = sorted(set(range(args.epoch_stride, args.epochs, args.epoch_stride))
                    | {1, args.epochs - 1})
    if args.reference_epochs:
        references = sorted(set(int(e) for e in args.reference_epochs.split(",")))
        if not references or not set(references).issubset(epochs):
            raise ValueError("Every reference epoch must belong to the exported positive epoch grid.")
    else:
        # Same dense-early / sparse-middle / dense-late selection as the aggregator.
        paper_grid = (set(range(1, min(100, args.epochs)))
                      | set(range(100, max(100, args.epochs - 20), 10))
                      | set(range(max(100, args.epochs - 20), args.epochs)))
        references = sorted((paper_grid & set(epochs)) | {epochs[-1]})
    layers = list(range(12))
    root = args.run_dir.resolve()
    cfg.update(
        backbone="vit", ind_name="imagenet", seed=f"_s{args.seed}", step=0,
        num_classes=1000, img_size=224, vit_size=768, num_heads=12,
        total_vit_epochs=args.epochs, pretrained_vit_IN1K=False,
        fixed_sae=False, trained_on="cls", train_split="train", eval_split="test",
        keywords=["train", "test"], train_patches=False, test_patches=False,
        combined=True, combined_images=True, concept_types=False,
        cache_activations=False, train_device=args.device, eval_device=args.device,
        activation_root=str(root / "activations"),
        vit_checkpoint_dir=str(args.checkpoint_dir.resolve()),
        data_dir=str(args.data_dir.resolve()),
        train_imglist=str(args.train_imglist.resolve()),
        test_imglist=str(args.test_imglist.resolve()),
        imglist_path=str(args.test_imglist.resolve()),
        log_path=str(root / "log.txt"),
    )
    cfg.pop("val_imglist", None)
    experiment_dir = root / "sae_checkpoints" / f"vit_imagenet_s{args.seed}"

    def for_reference(epoch, layer):
        name = f"refepoch_{epoch}_reflayer_{layer}"
        folder = experiment_dir / name
        return dict(cfg, ref_epoch=epoch, ref_layer=layer,
                    checkpoint_folder=str(folder), results_dir=str(folder / "eval_results"),
                    plots_dir=str(root / "figs" / f"vit_imagenet_s{args.seed}" / name))

    return cfg, epochs, references, layers, experiment_dir, for_reference


def target_epochs(reference, count, available):
    window = tracking.make_epoch_grid(reference, count, half_window=50, low_stride=10)
    # Exclude epoch 0, retain only exported epochs, and always include the reference.
    return sorted((set(map(int, window)) & set(available)) | {reference})


def check_exporter():
    source = inspect.getsource(activations.generate_acts_vit)
    loaders = inspect.getsource(activations.build_dataloaders)
    outputs = inspect.getsource(activations.get_model_outputs)
    if ("activation_root" not in source or "vit_checkpoint_dir" not in source
            or 'cfg.get("data_dir"' not in loaders
            or "cls_all.append(cls_layers.cpu())" not in outputs):
        raise RuntimeError("Apply the exporter edits in FULL_PIPELINE_SETUP.md before running export.")


def write_plot_log(source, destination, epochs):
    accuracy = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        match = re.search(r"epoch=(\d+)\s+.*?val_acc1=([0-9.]+)", line)
        if match:
            accuracy[int(match[1])] = float(match[2])
    missing = sorted(set(range(epochs)) - set(accuracy))
    if missing:
        raise ValueError(f"Training log is missing epoch records: {missing[:10]}")
    destination.write_text("".join(
        f"Epoch {e} Val Acc {accuracy[e]}\n" for e in range(epochs)
    ), encoding="utf-8")


def main(args=None):
    args = parse_args() if args is None else args
    cfg, epochs, references, layers, sae_dir, for_reference = experiment(args)
    root = args.run_dir.resolve()
    evo_dir = sae_dir / "evo_patterns"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the GPU environment and allocation for the full job.")
    for path in [args.data_dir, args.train_imglist, args.test_imglist]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.stage == "export" or (args.stage == "check" and not args.reuse_activations):
        check_exporter()
    if args.stage == "check" and args.reuse_activations:
        for epoch in epochs:
            folder = Path(cfg["activation_root"]) / f"vit_imagenet_s{args.seed}" / f"epoch{epoch}_step0"
            for split in ("train", "test"):
                for suffix in (".npy", "_y.npy", "_logits.npy"):
                    path = folder / f"vit_imagenet_{split}{suffix}"
                    if not path.is_file():
                        raise FileNotFoundError(path)
    root.mkdir(parents=True, exist_ok=True)
    # Stage restarts must use the same experiment settings and grids.
    manifest = dict(config=cfg, epochs=epochs, reference_epochs=references, layers=layers)
    manifest_path = root / "pipeline_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Run settings differ from pipeline_manifest.json; use a new run directory.")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Stage {args.stage}: {len(epochs)} checkpoints, {len(layers)} layers, "
          f"{len(references) * len(layers)} reference points", flush=True)
    if args.stage == "check":
        return
    if args.stage == "export":
        for epoch in epochs:
            # Exporter expects numeric seed; loader/checkpoint paths use '_sN'.
            activations.generate_acts_vit(dict(cfg, epoch=epoch, seed=args.seed))
        return
    if args.stage == "sae":
        ref = (epochs[-1], layers[-1])
        run_cfg = for_reference(*ref)
        tracking.train_sae_grid(
            run_cfg, epochs, layers, *ref,
            load_data=lambda e, l: tracking._load_all_data_cached(e, l, run_cfg, None),
            checkpoint_path=lambda e, l, r: tracking._checkpoint_path(run_cfg, e, l, r),
            trainer=train_sae_fast,
        )
        return
    if args.stage in {"evaluate", "plot"}:
        if args.stage == "plot":
            write_plot_log(args.training_log, Path(cfg["log_path"]), args.epochs)
        for epoch in references:
            selected = target_epochs(epoch, args.epochs, epochs)
            for layer in layers:
                run_cfg = for_reference(epoch, layer)
                config_dir = Path(run_cfg["results_dir"]) / "run_0"
                config_dir.mkdir(parents=True, exist_ok=True)
                (config_dir / "config.yaml").write_text(yaml.safe_dump(run_cfg), encoding="utf-8")
                path = evo_dir / f"evolution_patterns_e{epoch}_l{layer}.npz"
                if args.stage == "evaluate":
                    if path.exists():
                        with np.load(path) as saved:
                            if saved["epochs"].tolist() != selected:
                                raise ValueError(f"Existing epoch grid differs: {path}")
                        print("Existing result:", path, flush=True)
                    else:
                        tracking._evaluate_run_combined(0, run_cfg, selected, layers, epoch, layer, None)
                else:
                    tracking._save_concept_evo_pattern(
                        0, run_cfg, selected, layers, epoch, layer,
                        method="max", want_images=cfg.get("want_images", True), eval_device=args.device,
                    )
                plt.close("all")
        return
    if args.stage == "aggregate":
        scores, summary = aggregate.compute_scores_for_reference_grid(
            similarity_source=str(evo_dir), reference_epochs=references,
            reference_layers=layers, layer_values=layers, denominator="all",
            posterior="softmax", beta=12.0, onset_k=3, disappearance_k=5,
            start_end_fraction=0.05, stickiness_radius=1.0,
            reference_stickiness_radius=1.0, drift_threshold_layers=1.0,
            min_directional_efficiency=0.02, min_layer_localization=0.5,
            sticky_threshold=0.70, sticky_max_abs_drift_layers=0.50,
            eps=1e-12, max_width=4, choice="com",
        )
        scores.to_csv(root / "feature_scores.csv", index=False)
        summary.to_csv(root / "reference_summary.csv", index=False)
        aggregate.plot_aggregate_heatmaps(
            summary, metrics=[c for c in summary.columns
                              if c not in {"reference_epoch", "reference_layer"}],
            reference_epochs=references, reference_layers=layers,
            save_path=str(root / "heatmaps"), dpi=300, close_after_save=True,
            datasettag="ImageNet-1k",
        )


if __name__ == "__main__":
    main()
