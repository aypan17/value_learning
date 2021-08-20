#!/bin/bash

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
srun -w "$head_node" ray start --head --node-ip-address="$head_node_ip" --redis-port=$port \
    --num-cpus 4 --block & 

# __doc_script_start__
CONFIG=$1
EXP=$2
NAME=$3
REWARD=$4
WEIGHT=$5
WIDTH=$6
DEPTH=$7

if [ "${CONFIG}" = "test" ]; then
   python3 -u traffic_proxy.py ${EXP} test_${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} 4 --num_steps 2 --rollout_size 1 --horizon 300 --checkpoint 1 
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
    echo "Must select either 'ss' for short, single agent; 'ls' for long, single agent; 'sm' for short, multi agent; 'lm' for long, multi agent not ${CONFIG}"
    exit 0
fi 
