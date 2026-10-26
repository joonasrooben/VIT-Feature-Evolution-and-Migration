# ImageNet-1K Slurm pipeline

`run_full_pipeline.sbatch` runs the existing ImageNet training program followed by
activation export, SAE fitting, combined evaluation, feature plots and aggregation.
`run_full_pipeline.py` supplies command-line arguments to the existing stage functions.
Neither file reads or executes the tutorial notebook.

Use your GPU/FFCV environment for this job. `requirements_tutorial.txt` is the CPU
notebook environment, not the FFCV training environment. ImageNet images, the training
and test image lists, and the existing FFCV `.beton` files must be available on the
allocated node. Dataset acquisition and creation of the `.beton` files are prerequisites;
the repository does not contain the converter referenced in its old training launcher.

The launcher preserves the full trainer's current architecture: 12 blocks, width 192,
**12 attention heads**, and 1,000 classes. This differs from the five-block tutorial.
The trainer numbers saved checkpoints `0..299` for 300 epochs; checkpoint 0 already
contains one completed training epoch. As requested, the analysis grid excludes ID 0.
All layers `0..11` remain included. The default reference schedule and per-reference
epoch windows follow the existing aggregation and tracking scripts, with ID 0 removed.

## 1. Configure and submit

Activate your existing GPU/FFCV environment, then run from the repository root:

```bash
export DATA_DIR=/path/to/imagenet/images
export TRAIN_BETON=/path/to/train_raw_256.beton
export VAL_BETON=/path/to/val_raw_256.beton
export TRAIN_IMGLIST=/path/to/train_imagenet.txt
export TEST_IMGLIST=/path/to/test_imagenet.txt
export RUN_DIR=/path/to/shared/results/imagenet_s8
sbatch run_full_pipeline.sbatch
```

Image-list paths must resolve relative to `DATA_DIR`. Use persistent shared paths for
outputs, and adjust the sbatch partition, time and memory to the cluster. A fresh run
refuses to overwrite existing ViT checkpoints. Submission inherits your environment;
`PIPELINE_PYTHON=/absolute/path/to/python` can select its interpreter explicitly.

The file contains the training command and runs these stages as separate commands:

| Stage | Existing implementation |
| --- | --- |
| ViT training | `vit_tiny_imagenet_train_ffcv.py` |
| `export` | `generate_acts_vit_fun_tiny.generate_acts_vit` |
| `sae` | `sae_track_method.train_sae_grid`, with `train_sae_fast` |
| `evaluate` | `sae_track_method._evaluate_run_combined` |
| `plot` | `sae_track_method._save_concept_evo_pattern` |
| `aggregate` | `compute_scores_for_reference_grid`, `plot_aggregate_heatmaps` |

The launcher uses the YAML's SAE hyperparameters and matching settings. It sets the
dataset/model dimensions, paths, seed, devices and CLS-only mode for this ImageNet
experiment. Resolved settings and grids are saved in `pipeline_manifest.json`.
Each reference's resolved config is also saved under its normal `eval_results/run_0/`.
The FFCV log is converted into the existing plotter's `Val Acc` format; its parser
does not need another source change.

## 2. Reuse checkpoints or restart a stage

To start from existing trained backbones, provide their directory and complete log:

```bash
TRAIN_VIT=0 VIT_DIR=/path/to/checkpoints VIT_LOG=/path/to/training/log.out \
    sbatch run_full_pipeline.sbatch
```

To restart evaluation after the activation and SAE stages, retain the same `RUN_DIR`,
data paths, checkpoint directory, seed and settings:

```bash
TRAIN_VIT=0 START_STAGE=evaluate sbatch run_full_pipeline.sbatch
```

`START_STAGE` accepts `export`, `sae`, `evaluate`, `plot` or `aggregate`. A stage failure
stops the job; `set -o pipefail` preserves failures when output is sent through `tee`.
The existing `skip_existing` SAE setting is respected. Completed evolution files are
reused after checking their epoch axes. A changed experiment manifest requires a new
run directory, so incompatible outputs are not silently mixed.

Use `VIT_EPOCHS`, `VIT_BATCH_SIZE`, `VIT_WORKERS`, `EPOCH_STRIDE` and `REF_EPOCHS` to
control work. For example, `EPOCH_STRIDE=10 REF_EPOCHS=1,100,299` exports and fits epochs
1, 10, 20, ..., 290, 299 and evaluates those three reference epochs across all layers.
This is a reduced experiment, not the launcher's full default reference grid.

Outputs use the pipeline layout beneath `RUN_DIR`: `vit/`, `activations/`,
`sae_checkpoints/vit_imagenet_s8/evo_patterns/`, `figs/`, `heatmaps/` and `logs/`.
Aggregation also writes `feature_scores.csv` and `reference_summary.csv`.
