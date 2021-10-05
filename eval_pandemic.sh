#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=pandemic
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
#SBATCH -p 'jsteinhardt'
#SBATCH -w smaug

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/usr/local/linux/bin:/usr/local/bin:/usr/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

mkdir -p pandemic_policy_eval

EPOCH=$1
WIDTH=$2

#python3 pandemic_eval.py ~/pannoise/${EPOCH}_pan_noise_ ~/panpolicy/50_savio_panm_2layer_16 16 2 --width $WIDTH --epoch $EPOCH
python3 pandemic_eval.py ~/panpolicy/${EPOCH}_savio_panm_2layer_ ~/panpolicy/50_savio_panm_2layer_16 16 2 --width $WIDTH  --epoch $EPOCH

