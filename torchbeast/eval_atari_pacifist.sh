#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=atari
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
#SBATCH -p 'high' #'jsteinhardt'
# #SBATCH -w 'balrog'

FOLDER=$1
NAME=$2

export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/usr/local/linux/bin:/usr/local/bin:/usr/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin


python -m torchbeast.monobeast \
        --env RiverraidNoFrameskip-v4 \
        --mode eval \
        --savedir ${FOLDER} \
	--xpid ${NAME} \
        --num_episodes 4 \
	--fuel_multiplier -10 \

#python -m torchbeast.monobeast \
#        --env RiverraidNoFrameskip-v4 \
#        --mode test \
#        --savedir "./runs/pacifist" \
#        --num_episodes 4 \
#        --xpid pacifist_river50M_${NAME}_${FUEL} \
#        --hidden_size ${NAME} \
#        --num_layers ${FUEL} \
#        --fuel_multiplier -10
