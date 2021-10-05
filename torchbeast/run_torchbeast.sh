#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=atari
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH -p 'jsteinhardt'
#SBATCH -w 'balrog'
# #SBATCH -A fc_robustml
# #SBATCH -A co_stat

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/usr/local/linux/bin:/usr/local/bin:/usr/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

NAME=$1
FUEL=$2
WIDTH=$3
DEPTH=$4

python -m torchbeast.monobeast \
     --env RiverraidNoFrameskip-v4 \
     --num_actors 45 \
     --total_steps 50000000 \
     --learning_rate 0.0004 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 1024 \
     --unroll_length 100 \
     --num_buffers 60 \
     --num_threads 12 \
     --hidden_size ${WIDTH} \
     --num_layers ${DEPTH} \
     --fuel_multiplier ${FUEL} \
     --xpid ${NAME} \
     --savedir "~/torchbeast/runs" 
