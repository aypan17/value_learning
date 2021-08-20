#!/bin/bash
# shellcheck disable=SC2206
<<<<<<< HEAD
#SBATCH --job-name=pandemic
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH -p 'jsteinhardt'
#SBATCH -w balrog
=======
#SBATCH --time=72:00:00
#SBATCH --job-name=compare
# #SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
#SBATCH -p 'savio3'
#SBATCH -A fc_robustml
# #SBATCH -A co_stat
>>>>>>> savio

# set -x

# simulate conda activate flow
export PATH=/global/home/users/aypan17/sumo/bin:/global/home/users/aypan17/redis/src:/global/home/users/aypan17/cmake/bin:/global/home/groups/co_stat/software/miniconda3_aypan17/envs/pandemic/bin:/global/home/groups/co_stat/software/miniconda3_aypan17/condabin:/global/home/groups/consultsw/sl-7.x86_64/modules/sq/0.1.0/bin:/global/software/sl-7.x86_64/modules/tools/emacs/25.1/bin:/global/software/sl-7.x86_64/modules/tools/vim/7.4/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/global/home/groups/allhands/bin:/global/home/users/aypan17/bin

# Move wandb logs to scratch 
export WANDB_DIR=/global/scratch/aypan17/ 

mkdir -p pandemic_policy

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

<<<<<<< HEAD
NAME=$1
ALPHA=$2
BETA=$3
GAMMA=$4
DELTA=$5
LO=$6
HI=$7
WIDTH=$8
DEPTH=$9
DISCOUNT=${10}

if [ "${NAME}" = "sacd" ]; then
	python3 pandemic_test.py $NAME $ALPHA $BETA $GAMMA $DELTA $LO $HI 0 0 $DISCOUNT --n_cpus "$SLURM_CPUS_PER_TASK" --sacd
elif [ "${NAME}" = "test" ]; then
	python3 pandemic_test.py $NAME 0 0 0 0 95 105 32 3 0.99 --n_cpus "$SLURM_CPUS_PER_TASK" --test
else
	python3 pandemic_test.py $NAME $ALPHA $BETA $GAMMA $DELTA $LO $HI $WIDTH $DEPTH $DISCOUNT --n_cpus "$SLURM_CPUS_PER_TASK" 
fi
=======
mkdir -p pandemic_policy

export SLURM_CPUS_PER_TASK=$SLURM_CPUS_ON_NODE

python3 pandemic_test.py "$SLURM_CPUS_PER_TASK" 0.4 1 0.1 0.02
>>>>>>> savio
