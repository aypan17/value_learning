#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=atari
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
# #SBATCH -p 'high' #'jsteinhardt'
# #SBATCH -w 'balrog'

NAME=$1
MOVE=$2
TMOVE=$3
WIDTH=$4
DEPTH=$5

export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/usr/local/linux/bin:/usr/local/bin:/usr/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

python -m torchbeast.monobeast \
        --env RiverraidNoFrameskip-v4 \
        --mode test_render \
        --savedir "./runs" \
        --num_episodes 1 \
        --xpid $NAME \
        --hidden_size $WIDTH \
        --num_layers $DEPTH \
        --move_penalty $MOVE \
	--true_move_penalty $TMOVE

