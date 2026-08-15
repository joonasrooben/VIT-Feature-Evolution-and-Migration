#!/usr/bin/bash
#SBATCH -J due
#SBATCH --partition=gpu
#SBATCH --nodelist=firefly[1-2],pegasus,pegasus2

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

#SBATCH --array=0-348%10


set -euo pipefail

export SCIPY_ARRAY_API=1
##360
##RE_VALUES=($(seq 98 100) $(seq 110 10 280) $(seq 281 299))
##RE_VALUES=($(seq 281 299))
RE_VALUES=($(seq 72 100))
RL_VALUES=($(seq 0 11))


NUM_RE=${#RE_VALUES[@]}
NUM_RL=${#RL_VALUES[@]}
TOTAL=$((NUM_RE * NUM_RL))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "$TASK_ID" -ge "$TOTAL" ]; then
    echo "TASK_ID=$TASK_ID exceeds TOTAL=$TOTAL"
    exit 1
fi

# ---------------------------------------------------------
# Map one array index to one (re, rl) pair
# ---------------------------------------------------------
RE_INDEX=$((TASK_ID / NUM_RL))
RL_INDEX=$((TASK_ID % NUM_RL))

RE=${RE_VALUES[$RE_INDEX]}
RL=${RL_VALUES[$RL_INDEX]}

echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
echo "Running with --re $RE --rl $RL"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -u 2d_sae_track_independent_shapes3d.py \
    --re "$RE" \
    --rl "$RL" \

##--config sae_exp_config_base_s7.yml 

