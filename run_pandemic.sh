#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=pandemic
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH -p 'jsteinhardt'
#SBATCH -w smaug

# set -x

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/usr/local/linux/bin:/usr/local/bin:/usr/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

mkdir -p pandemic_policy

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

CONFIG=$1
NAME=$2
ALPHA=$3
BETA=$4
GAMMA=$5
DELTA=$6
LO=$7
HI=$8
WIDTH=$9
DEPTH=${10}
DISCOUNT=${11}

if [ "${CONFIG}" = "sacd" ]; then
	python3 pandemic_test.py $NAME $ALPHA $BETA $GAMMA $DELTA $LO $HI 0 0 $DISCOUNT --n_cpus "$SLURM_CPUS_PER_TASK" --sacd
elif [ "${CONFIG}" = "runtime" ]; then
	python3 runtime.py
	exit  0
elif [ "${CONFIG}" = "test" ]; then
	python3 pandemic_test.py 'test' 0 0 0 0 95 105 32 3 0.99 --n_cpus "$SLURM_CPUS_PER_TASK" --test --four_start
elif [ "${CONFIG}" = "run" ]; then
	python3 pandemic_test.py $NAME $ALPHA $BETA $GAMMA $DELTA $LO $HI $WIDTH $DEPTH $DISCOUNT --n_cpus "$SLURM_CPUS_PER_TASK" 
elif [ "${CONFIG}" = "four" ]; then
	python3 pandemic_test.py $NAME $ALPHA $BETA $GAMMA $DELTA $LO $HI $WIDTH $DEPTH $DISCOUNT --n_cpus "$SLURM_CPUS_PER_TASK" --four_start 
else
	echo "Invalid config"
	exit 0
fi
