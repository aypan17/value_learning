#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=fin
#SBATCH --cpus-per-task=10
# #SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH -p 'gpu_jsteinhardt'
#SBATCH -w shadowfax

set -x 

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/usr/local/cuda-11.1/bin:/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/bin:/usr/local/linux/bin:/usr/bin:/usr/local/bin:/usr/X11R6/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

# Move wandb logs to scratch 
export WANDB_DIR=/global/scratch/aypan17/ 

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

MODE=$1
VOL=$2
TVOL=$3
MORAL=$4
ENV=$5
SOCIAL=$6
NAME=$7
CONFIG=$8


if [ "${MODE}" = "test" ]; then
	python3 fin_misweight.py 0 0 0 "$SLURM_CPUS_PER_TASK" --rollout_size 64 --num_steps 250 --bs 64 --eval_freq 250 --state_date "2015-01-01" --mid_date "2016-01-01" --end_date "2017-01-01"
	exit 0 
fi

if [ "${CONFIG}" = "s" ]; then
	python3 fin_${MODE}.py $MORAL $ENV $SOCIAL "$SLURM_CPUS_PER_TASK" --save_path $NAME --num_steps 500000 --bs 1024 --vol_multiplier $VOL --true_vol_multiplier $TVOL
elif [ "${CONFIG}" = "m" ]; then
	python3 fin_${MODE}.py $MORAL $ENV $SOCIAL "$SLURM_CPUS_PER_TASK" --save_path $NAME --rollout_size 256 --num_steps 1000000 --bs 1024 --vol_multiplier $VOL --true_vol_multiplier $TVOL
elif [ "${CONFIG}" = "l" ]; then
	python3 fin_${MODE}.py $MORAL $ENV $SOCIAL "$SLURM_CPUS_PER_TASK" --save_path $NAME --vol_multiplier $VOL --true_vol_multiplier $TVOL
else
	echo "Invalid config"
	exit 0
fi
