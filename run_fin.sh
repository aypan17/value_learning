#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=horizon3000
#SBATCH --cpus-per-task=10
# #SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH -p 'gpu_jsteinhardt'
# #SBATCH -w smaug-gpu 

set -x 

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/accounts/projects/jsteinhardt/aypan/value_learning/bin:/bin:/usr/local/linux/bin:/usr/bin:/usr/local/bin:/usr/X11R6/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

MODE=$1
MORAL=$2
ENV=$3
SOCIAL=$4
NAME=$5
CONFIG=$6


if [ "${MODE}" = "test" ]; then
	python3 fin_misweight.py 0 0 0 "$SLURM_CPUS_PER_TASK" --rollout_size 64 --num_steps 250 --bs 64 --eval_freq 250 --state_date "2019-01-01" --mid_date "2020-01-01" --end_date "2021-01-01"
	exit 0 
fi

if [ "${CONFIG}" = "s" ]; then
	python3 fin_${MODE}.py $MORAL $ENV $SOCIAL "$SLURM_CPUS_PER_TASK" --save_path $NAME --rollout_size 256 --num_steps 250000 --bs 256 
elif [ "${CONFIG}" = "l" ]; then
	python3 fin_${MODE}.py $MORAL $ENV $SOCIAL "$SLURM_CPUS_PER_TASK" --save_path $NAME
else
	echo "Invalid config"
	exit 0
fi

