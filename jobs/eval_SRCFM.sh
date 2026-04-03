#!/bin/bash -l
#SBATCH --chdir=/share/home/okabayas/super-res-fm/runner
#SBATCH --gres gpu:1
#SBATCH --constraint a100
#SBATCH --cpus-per-gpu 16
#SBATCH --mem-per-cpu 6G
#SBATCH --partition longrun
#SBATCH --output job_logs/eval.out
#SBATCH --time=4-00:00:00

setcuda 12.1
conda activate test

### OPTIONAL, copy data project data  --nproc_per_node=2  ###

torchrun src/eval.py trainer=gpu ckpt_path="/share/home/okabayas/super-res-fm/runner/logs/train/runs/RGBNIR_orasis/checkpoints/orasis.ckpt" 
