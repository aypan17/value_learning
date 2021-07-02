#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=horizon3000
#SBATCH --cpus-per-task=20
# #SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
#SBATCH -p 'gpu_jsteinhardt'
# #SBATCH -w smaug-gpu 

set -x 

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/bin:/usr/local/linux/bin:/usr/bin:/usr/local/bin:/usr/X11R6/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

python3 train.py "$SLURM_CPUS_PER_TASK"

