#!/bin/bash -l
#SBATCH --chdir=/share/home/okabayas/super-res-fm/runner
#SBATCH --gres=gpu:1
#SBATCH --constraint a6000
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 16G
#SBATCH --partition longrun
#SBATCH --output job_logs/train_locenc.out
#SBATCH --time=7-00:00:00

setcuda 12.1
conda activate test

### OPTIONAL, copy data project data ###

python src/train.py trainer=gpu ckpt_path="/share/home/okabayas/SR-flow-matching/runner/logs/train/runs/2025-04-09_13-34-53/checkpoints/last.ckpt" 
