#!/bin/bash
#SBATCH --time=10:00:00   # walltime
# #SBATCH --ntasks=1	 # number of processor cores (i.e. tasks)
#SBATCH -n 12	# number of nodes
#SBATCH --qos normal #QOS to run in
#SBATCH --gres gpu:0 # number of GPU
#SBATCH --mem-per-cpu=16G	# memory per CPU core
#SBATCH -J %x  # job name

# #SBATCH --mail-type=ALL						
# #SBATCH --mail-user=aypan@caltech.edu

ENV_NAME=$1
MODE=$2
EXP_NAME=$3

valid_envs="traffic"
valid_modes="train test"

[[ $valid_envs =~ (^|[[:space:]])"$ENV_NAME"($|[[:space:]]) && $valid_modes =~ (^|[[:space:]])"$MODE"($|[[:space:]]) ]] && python3 ${ENV_NAME}_${MODE}.py $EXP_NAME || echo "The environment '${ENV_NAME}' is not one of [${valid_envs}] or the mode '${MODE}' is not one of [${valid_modes}]" 

