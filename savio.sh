#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --time=00:15:00
#SBATCH --job-name=compare
# #SBATCH --cpus-per-task=6
# #SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
# #SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
#SBATCH -p 'savio3'
# #SBATCH -A fc_robustml
#SBATCH -A co_stat

# set -x

# load flow environment from conda and load GNU parallel
export PATH=/global/home/users/aypan17/sumo/bin:/global/home/users/aypan17/redis/src:/global/home/users/aypan17/cmake/bin:/global/home/groups/consultsw/sl-7.x86_64/modules/sq/0.1.0/bin:/global/software/sl-7.x86_64/modules/tools/gnu-parallel/2019.03.22/bin:/global/home/users/aypan17/sumo/bin:/global/home/users/aypan17/cmake/bin:/global/home/groups/consultsw/sl-7.x86_64/modules/sq/0.1.0/bin:/global/home/users/aypan17/sumo/bin:/global/home/users/aypan17/cmake/bin:/global/home/groups/co_stat/software/miniconda3_aypan17/envs/flow/bin:/global/home/groups/co_stat/software/miniconda3_aypan17/condabin:/global/home/groups/consultsw/sl-7.x86_64/modules/sq/0.1.0/bin:/global/software/sl-7.x86_64/modules/tools/emacs/25.1/bin:/global/software/sl-7.x86_64/modules/tools/vim/7.4/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/global/home/groups/allhands/bin:/global/home/users/aypan17/bin:/global/home/groups/allhands/bin:/global/home/groups/allhands/bin

CONFIG=$1
EXP=$2
NAME=$3
REWARD=$4
WEIGHT=$5
valid="ss ls sm lm test"

WIDTHS=(4 16 64 256)
DEPTHS=(3 3 3 3)
JOBS=4
CPUS_PER_TASK=$(( $SLURM_CPUS_ON_NODE / $JOBS )) 

srun="srun"
parallel="parallel --delay .2 --link --jobs $JOBS --resume --joblog parallel-${SLURM_JOB_ID}.log"

#$parallel  ./test.sh {1}  ::: "${WIDTHS[@]}"
#exit 0

[[ " $valid " =~ " ${CONFIG} " ]] && $parallel "./run_savio_traffic.sh ${CONFIG} ${EXP} ${NAME}_{1}_{2} ${REWARD} ${WEIGHT} {1} {2}" ::: "${WIDTHS[@]}" ::: "${DEPTHS[@]}" || echo "must select either 'ss' for short, single agent; 'ls' for long, single agent; 'sm' for short, multi agent; 'lm' for long, multi agent not '${CONFIG}'"
exit 0

if [ "${EXP}" = "test" ]; then
    parallel --link --jobs $JOBS_PER_NODE python3 -u traffic_proxy.py singleagent_merge "test" vel,accel 1,20 {1} {2} "$SLURM_CPUS_PER_TASK" --num_steps 2 --rollout_size 1 --horizon 300 --checkpoint 1 ::: 4 16 64 256 ::: 3 3 3 3
    exit 0 
fi

if [ "${CONFIG}" = "ss" ]; then
    python3 -u traffic_proxy.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} "$SLURM_CPUS_PER_TASK" --num_steps 5000 --rollout_size 7 --horizon 300 
elif [ "${CONFIG}" = "ls" ]; then
    python3 -u traffic_proxy.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} "$SLURM_CPUS_PER_TASK" 
elif [ "${CONFIG}" = "sm" ]; then
    python3 -u traffic_proxy.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} "$SLURM_CPUS_PER_TASK"  --num_steps 5000 --rollout_size 7 --horizon 300 --multi
elif [ "${CONFIG}" = "lm" ]; then
    python3 -u traffic_proxy.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} "$SLURM_CPUS_PER_TASK" --multi 
else
    echo "must select either 'ss' for short, single agent; 'ls' for long, single agent; 'sm' for short, multi agent; 'lm' for long, multi agent not ${config}"
    exit 0
fi 
