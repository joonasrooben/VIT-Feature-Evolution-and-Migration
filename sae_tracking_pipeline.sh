#!/usr/bin/bash

GPU_JOB_ID=$(sbatch --parsable sae_tracking.sh)

echo "Submitted GPU job: ${GPU_JOB_ID}"

CPU_JOB_ID=$(sbatch --parsable --dependency=afterok:${GPU_JOB_ID} sae_tracking_figs.sh)

echo "Submitted CPU job: ${CPU_JOB_ID}"