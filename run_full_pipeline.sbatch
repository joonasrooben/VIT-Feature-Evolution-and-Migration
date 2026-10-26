#!/usr/bin/env bash
#SBATCH --job-name=vit-sae-imagenet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=100:00:00
#SBATCH --output=vit-sae-imagenet-%j.out
#SBATCH --error=vit-sae-imagenet-%j.err

# ImageNet-1K: ViT training -> activation export -> SAE training -> combined
# evaluation -> pattern plots -> aggregate tables/heatmaps. No notebook execution.
# Apply FULL_PIPELINE_SETUP.md first. Activate your GPU + FFCV environment before
# submitting; requirements_tutorial.txt is for the CPU tutorial only.
#
# From this repository (override paths below through exported environment variables):
#   sbatch run_full_pipeline.sbatch
# Reuse existing ViT checkpoints and their original training log:
#   TRAIN_VIT=0 VIT_DIR=/path/to/checkpoints VIT_LOG=/path/to/log.out sbatch run_full_pipeline.sbatch
# Restart later stages using the SAME run directory/settings:
#   START_STAGE=evaluate RUN_DIR=/path/to/previous/run TRAIN_VIT=0 sbatch run_full_pipeline.sbatch
# For fewer exported/fitted checkpoints: EPOCH_STRIDE=10 (final checkpoint included).
# REF_EPOCHS=1,100,299 optionally restricts references; each must be exported.
# Adapt the partition, memory and time to your cluster and dataset size.

set -euo pipefail
cd "${PIPELINE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PIPELINE_PYTHON:-python}"
export RUN_DIR="${RUN_DIR:-$PWD/results/full_pipeline_${SLURM_JOB_ID:-local}}"
VIT_DIR="${VIT_DIR:-$RUN_DIR/vit}"
VIT_LOG="${VIT_LOG:-$VIT_DIR/log.out}"
DATA_DIR="${DATA_DIR:-/tmp/imagenet_1k}"
TRAIN_BETON="${TRAIN_BETON:-$DATA_DIR/train_raw_256.beton}"
VAL_BETON="${VAL_BETON:-$DATA_DIR/val_raw_256.beton}"
TRAIN_IMGLIST="${TRAIN_IMGLIST:-/gpfs/space/projects/mlgroup/data/benchmark_imglist/imagenet/train_imagenet.txt}"
TEST_IMGLIST="${TEST_IMGLIST:-/gpfs/space/projects/mlgroup/data/benchmark_imglist/imagenet/test_imagenet.txt}"
SAE_CONFIG="${SAE_CONFIG:-$PWD/sae_exp_config_base.yml}"
VIT_EPOCHS="${VIT_EPOCHS:-300}"
VIT_SEED="${VIT_SEED:-8}"
TRAIN_VIT="${TRAIN_VIT:-1}"
START_STAGE="${START_STAGE:-export}"
export SCIPY_ARRAY_API=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

stages=(export sae evaluate plot aggregate)
start_index=-1
for i in "${!stages[@]}"; do
    if [[ "${stages[$i]}" == "$START_STAGE" ]]; then start_index=$i; fi
done
if (( start_index < 0 )); then
    echo "START_STAGE must be export, sae, evaluate, plot or aggregate." >&2
    exit 1
fi
if [[ "$TRAIN_VIT" != 0 && "$TRAIN_VIT" != 1 ]]; then
    echo "TRAIN_VIT must be 0 or 1." >&2
    exit 1
fi
if (( start_index > 0 )) && [[ "$TRAIN_VIT" == 1 ]]; then
    echo "A later-stage restart requires TRAIN_VIT=0 to preserve existing activations." >&2
    exit 1
fi
mkdir -p "$RUN_DIR/logs"
common=(--config "$SAE_CONFIG" --run-dir "$RUN_DIR" --checkpoint-dir "$VIT_DIR"
        --data-dir "$DATA_DIR" --train-imglist "$TRAIN_IMGLIST" --test-imglist "$TEST_IMGLIST"
        --training-log "$VIT_LOG" --epochs "$VIT_EPOCHS" --seed "$VIT_SEED"
        --epoch-stride "${EPOCH_STRIDE:-1}" --reference-epochs "${REF_EPOCHS:-}" --device cuda)
if (( start_index > 0 )); then common+=(--reuse-activations); fi

# Check configuration, paths, GPU availability and exporter edits BEFORE training.
srun "$PYTHON_BIN" -u run_full_pipeline.py "${common[@]}" --stage check

# 1. Train the repository's 12-block, width-192, 12-head ImageNet model.
if [[ "$TRAIN_VIT" == 1 ]]; then
    [[ -f "$TRAIN_BETON" ]] || { echo "Missing $TRAIN_BETON" >&2; exit 1; }
    [[ -f "$VAL_BETON" ]] || { echo "Missing $VAL_BETON" >&2; exit 1; }
    if compgen -G "$VIT_DIR/epoch_*.ckpt" > /dev/null; then
        echo "Existing ViT checkpoints in $VIT_DIR. Use TRAIN_VIT=0 or a new directory." >&2
        exit 1
    fi
    mkdir -p "$VIT_DIR" "$(dirname "$VIT_LOG")"
    srun "$PYTHON_BIN" -u vit_tiny_imagenet_train_ffcv.py \
        --train-beton "$TRAIN_BETON" --val-beton "$VAL_BETON" \
        --train-samples "${TRAIN_SAMPLES:-1281167}" --val-samples "${VAL_SAMPLES:-50000}" \
        --num-classes 1000 --output-dir "$VIT_DIR" --epochs "$VIT_EPOCHS" --seed "$VIT_SEED" \
        --batch-size "${VIT_BATCH_SIZE:-2048}" --num-workers "${VIT_WORKERS:-30}" \
        --channels-last 2>&1 | tee "$VIT_LOG"
fi

# 2. generate_acts_vit: export .npy representations, labels and logits.
# 3. train_sae_grid: fit one SAE for every positive checkpoint/layer pair.
# 4. _evaluate_run_combined: write each reference's evolution_patterns_*.npz.
# 5. _save_concept_evo_pattern: reload those files and plot features/images.
# 6. compute_scores_for_reference_grid + plot_aggregate_heatmaps: CSVs and figures.
# Each command runs a separate stage directly, with its own log and exit status.
for ((i=start_index; i<${#stages[@]}; i++)); do
    stage="${stages[$i]}"
    srun "$PYTHON_BIN" -u run_full_pipeline.py "${common[@]}" --stage "$stage" \
        2>&1 | tee "$RUN_DIR/logs/$stage.log"
done
echo "Full ImageNet pipeline completed: $RUN_DIR"
