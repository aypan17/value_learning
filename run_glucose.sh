#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=glucose
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH -p 'jsteinhardt'
#SBATCH -w shadowfax
# set -x 

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/glucose/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/usr/local/linux/bin:/usr/local/bin:/usr/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

NAME=$1
WIDTH=$2
PROXY=$3
TRUE=$4

if [ "${NAME}" = "test" ]; then
	python3 glucose_rlkit.py $NAME 4 'magni_bg_insulin' 'magni_bg_insulin_true' $SLURM_CPUS_PER_TASK 'True'
else
	python3 glucose_rlkit.py $NAME $WIDTH $PROXY $TRUE $SLURM_CPUS_PER_TASK 'False'
fi

