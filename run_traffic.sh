#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=compare
#SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
# #SBATCH -w shadowfax
# #SBATCH -p 'high_pre'

# set -x

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/usr/local/cuda-11.1/bin:/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/bin:/usr/local/linux/bin:/usr/bin:/usr/local/bin:/usr/X11R6/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=$(($RANDOM + 1024))
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --redis-port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block & # --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
# __doc_head_ray_end__


# __doc_script_start__
EXP=$1
NAME=$2
REWARD=$3
WEIGHT=$4
WIDTH=$5
DEPTH=$6
CONFIG=$7

if [ "${EXP}" = "test" ]; then
    python3 -u traffic_savio.py singleagent_merge_bus "test" delay,still 1,0.2 32 3 --n_cpus "$SLURM_CPUS_PER_TASK" --num_steps 3 --rollout_size 4 --horizon 300 --checkpoint 1 --test
    exit 0 
fi

if [ "${CONFIG}" = "ss" ]; then
    python3 -u traffic_savio.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} --n_cpus "$SLURM_CPUS_PER_TASK" --num_steps 5000 --rollout_size 7 --horizon 300 
elif [ "${CONFIG}" = "ls" ]; then
    python3 -u traffic_savio.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} --n_cpus "$SLURM_CPUS_PER_TASK" 
elif [ "${CONFIG}" = "sm" ]; then
    python3 -u traffic_savio.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} --n_cpus "$SLURM_CPUS_PER_TASK"  --num_steps 5000 --rollout_size 7 --horizon 300 --multi
elif [ "${CONFIG}" = "lm" ]; then
    python3 -u traffic_savio.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} --n_cpus "$SLURM_CPUS_PER_TASK" --multi 
else
    echo "Must select either 'ss' for short, single agent; 'ls' for long, single agent; 'sm' for short, multi agent; 'lm' for long, multi agent not ${CONFIG}"
    exit 0
fi 
