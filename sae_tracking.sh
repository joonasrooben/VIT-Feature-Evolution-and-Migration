#!/usr/bin/bash

#The name of the job is test_job
#SBATCH -J due 

#The job requires 1 compute node
#SBATCH -N 1
##SBATCH --partition=main

##SBATCH --partition=gpu
##SBATCH --nodelist=firefly1,firefly2,pegasus,pegasus2
#SBATCH --nodes=1
##SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=1
#SBACTH --hint=nomultithread
#SBATCH --mem=6GB

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is a half hour
#SBATCH -t 10:00:00

export SCIPY_ARRAY_API=1
python -u 2d_sae_track_independent_shapes3d.py --re 299 --rl 11 ##--config sae_exp_config_base_s7.yml 
##python -u evo_aggregate_patterns.py
##'./sae_checkpoints/vit_imagenet_mixed10_balanced_s7/refepoch_49_reflayer_11/eval_results/config.yaml'





