#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=pandemic
#SBATCH --cpus-per-task=1
# #SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
#SBATCH -p 'high'
# #SBATCH -w shadowfax

set -x 

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/usr/local/cuda-11.1/bin:/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/bin:/usr/local/linux/bin:/usr/bin:/usr/local/bin:/usr/X11R6/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

mkdir -p pandemic_policy

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

NAME=$1
ALPHA=$2
BETA=$3
GAMMA=$4
DELTA=$5
DISCOUNT=$6
WIDTH=$7
DEPTH=$8

if [ "${NAME}" = "sacd" ]; then
	python3 pandemic_test.py $NAME $ALPHA $BETA $GAMMA $DELTA $DISCOUNT --n_cpus "$SLURM_CPUS_PER_TASK" --sacd
else
	python3 pandemic_test.py $NAME $ALPHA $BETA $GAMMA $DELTA $DISCOUNT $WIDTH $DEPTH "$SLURM_CPUS_PER_TASK" 
fi
