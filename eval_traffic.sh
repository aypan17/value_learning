#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=traffic
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
#SBATCH -p 'jsteinhardt' 
#SBATCH -w balrog

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/usr/local/linux/bin:/usr/local/bin:/usr/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

BASELINE=$1
RESULTS=$8
PROXY=$2
PWEIGHTS=$3
TRUE=$4
TWEIGHTS=$5
TRUE2=$6
T2WEIGHTS=$7

python3 flow/utils/compute_norms.py $RESULTS $PROXY $PWEIGHTS $TRUE $TWEIGHTS --true2 $TRUE2 --true2_weights $T2WEIGHTS --baseline $BASELINE 
