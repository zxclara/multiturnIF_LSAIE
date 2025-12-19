#!/bin/bash
#SBATCH --job-name=sft_multi_if_serve
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:59:59
#SBATCH --output=/iopsstor/scratch/cscs/yanghao/project/log/sft_eval/mco_scaled_half_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/yanghao/project/log/sft_eval/mco_scaled_half_%j.err
#SBATCH --environment=/iopsstor/scratch/cscs/yanghao/project/env/vllm_env.toml

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

# The sbatch script is executed by only one node.
echo "[sbatch-master] running on $(hostname)"

echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

echo "[sbatch-master] define some env vars that will be passed to the compute nodes"

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
export MASTER_PORT=6017  # Choose an unused port
export FOOBAR=666
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

# print master IP address to be used by requests
export MASTER_IP=$(hostname -i)
echo "[sbatch-master] MASTER_IP: ${MASTER_IP}"

export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HF_HOME=/capstor/scratch/cscs/${USER}/models/
export HF_HUB_CACHE=/capstor/scratch/cscs/${USER}/models/

echo "[sbatch-master] execute command on compute nodes"

# The command that will run on each node
VLLM_CMD="
vllm serve \
    --dtype auto \
    --pipeline-parallel-size ${SLURM_NNODES} \
    --config /iopsstor/scratch/cscs/"${USER}"/project/configs/serving/qwen3_8b_mco_scaled_half.yaml
"
CMD="
# print current environment variables
echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

# print vllm command
echo \"VLLM_CMD: ${VLLM_CMD}\"

# run the script at runtime
bash -c '${VLLM_CMD}'
"

srun bash -c "$CMD"

echo "[sbatch-master] task finished"