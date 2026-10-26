#!/usr/bin/bash

#The name of the job is test_job
#SBATCH -J due 

#The job requires 1 compute node
#SBATCH -N 1

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --nodelist=pegasus2

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBACTH --hint=nomultithread

#The maximum walltime of the job is a half hour
#SBATCH -t 100:00:00


set -euo pipefail


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python vit_tiny_imagenet_train_ffcv.py \
  --train-beton /tmp/imagenet_1k/train_raw_256.beton \
  --val-beton /tmp/imagenet_1k/val_raw_256.beton \
  --train-samples 1281167 \
  --val-samples 50000 \
  --output-dir ./results/vit_tiny_ffcv \
  --batch-size 2048 \
  --num-workers 30 \
  --channels-last


#python build_imagenet_beton_from_imglist.py \
#  --data-root /tmp/imagenet_1k \
#  --imglist /gpfs/space/projects/mlgroup/data/benchmark_imglist/imagenet/train_imagenet.txt \
#  --output /tmp/imagenet_1k/train_raw_256.beton \
#  --write-mode raw \
#  --max-resolution 256 \
#  --num-workers 16

#python build_imagenet_beton_from_imglist.py \
#  --data-root /tmp/imagenet_1k \
#  --imglist /gpfs/space/projects/mlgroup/data/benchmark_imglist/imagenet/test_imagenet.txt \
#  --output /tmp/imagenet_1k/val_raw_256.beton \
#  --write-mode raw \
#  --max-resolution 256 \
#  --num-workers 16

#python generate_acts_vit.py --ind_name shapes3d --epoch 49 --step 0 --num_classes 4 --seed 2 --total_epochs 50 --vit_size 768 --num_heads 3
#python generate_acts_vit.py --ind_name shapes3d --epoch 45 --step 0 --num_classes 4 --seed 2 --total_epochs 50 --vit_size 768 --num_heads 3
#python generate_acts_vit.py --ind_name shapes3d --epoch 15 --step 0 --num_classes 3 --seed 3 --total_epochs 150 --vit_size 768 --num_heads 3
#python generate_acts_vit.py --ind_name shapes3d --epoch 150 --step 0 --num_classes 3 --seed 3 --total_epochs 150 --vit_size 768 --num_heads 3


#yes | python -u main.py \
#    --config configs/datasets/imagenet200/imagenet_mixed10_balanced.yml \
#    configs/networks/vit.yml \
#    configs/pipelines/train/vit_tiny_a1.yml\
#    --seed 11







