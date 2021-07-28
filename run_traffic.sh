#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --time=72:00:00
#SBATCH --job-name=compare
#SBATCH --cpus-per-task=40
# #SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
#SBATCH -p 'savio3'
#SBATCH -A fc_robustml
# #SBATCH -A co_stat

# set -x

# simulate conda activate flow
export PATH=/global/home/users/aypan17/sumo/bin:/global/home/users/aypan17/cmake/bin:/global/home/groups/consultsw/sl-7.x86_64/modules/sq/0.1.0/bin:/global/home/users/aypan17/cmake/bin:/global/home/groups/consultsw/sl-7.x86_64/modules/sq/0.1.0/bin:/global/home/users/aypan17/ninja:/global/home/users/aypan17/cmake/bin:/global/home/groups/consultsw/sl-7.x86_64/modules/sq/0.1.0/bin:/global/home/users/aypan17/cmake/bin:/global/home/groups/consultsw/sl-7.x86_64/modules/sq/0.1.0/bin:/global/home/users/aypan17/cmake/bin:/global/home/groups/consultsw/sl-7.x86_64/modules/sq/0.1.0/bin:/global/home/groups/co_stat/software/miniconda3_aypan17/envs/flow/bin:/global/home/groups/co_stat/software/miniconda3_aypan17/condabin:/global/home/groups/consultsw/sl-7.x86_64/modules/sq/0.1.0/bin:/global/software/sl-7.x86_64/modules/tools/emacs/25.1/bin:/global/software/sl-7.x86_64/modules/tools/vim/7.4/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/global/home/groups/allhands/bin:/global/home/users/aypan17/bin:/global/home/groups/allhands/bin:/global/home/groups/allhands/bin:/global/home/groups/allhands/bin:/global/home/groups/allhands/bin:/global/home/groups/allhands/bin

# Move wandb logs to scratch 
export WANDB_DIR=/global/scratch/aypan17/ 

# Neptune
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4MDkxNjhkMi00ZWZkLTQ0OWQtYTgzOS1iNTcxN2ZkYWZjOWYifQ=="

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

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 3

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done
# __doc_worker_ray_end__

# __doc_script_start__
CONFIG=$1
EXP=$2
NAME=$3
REWARD=$4
WEIGHT=$5

if [ "${CONFIG}" = "test" ]; then
    python3 -u traffic_savio.py singleagent_bottleneck "test" vel,accel 1,20 --num_steps 2 --rollout_size 7 --horizon 300 --checkpoint 1 
    exit 0 
fi

if [ "${CONFIG}" = "ss" ]; then
    python3 -u traffic_savio.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} --num_steps 5000 --rollout_size 7 --horizon 300 
elif [ "${CONFIG}" = "ls" ]; then
    python3 -u traffic_savio.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} --rollout_size 7 
elif [ "${CONFIG}" = "sm" ]; then
    python3 -u traffic_savio.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} --num_steps 5000 --rollout_size 7 --horizon 300 --multi
elif [ "${CONFIG}" = "lm" ]; then
    python3 -u traffic_savio.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} --rollout_size 7 --multi 
else
    echo "Must select either 'ss' for short, single agent; 'ls' for long, single agent; 'sm' for short, multi agent; 'lm' for long, multi agent not ${CONFIG}"
    exit 0
fi 
