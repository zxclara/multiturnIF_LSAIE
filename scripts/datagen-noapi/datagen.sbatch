#!/bin/bash
#SBATCH --job-name=sft_multi_if
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --time=00:12:00
#SBATCH --output=/iopsstor/scratch/cscs/yanghao/project/log/datagen/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/yanghao/project/log/datagen/%j.err
#SBATCH --environment=/iopsstor/scratch/cscs/yanghao/project/env/if_multi_sft.toml

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
export MASTER_PORT=12345   # Choose an unused port
export FOOBAR=666
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

echo "[sbatch-master] execute command on compute nodes"

# The command that will run on each node
CMD="
# print current environment variables
echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

# unset SSL_CERT_FILE
unset SSL_CERT_FILE

# run the script
python /iopsstor/scratch/cscs/"${USER}"/project/datagen/async_requests.py \
    --model=Qwen/Qwen3-Next-80B-A3B-Thinking \
    --swissai_api=<YOUR_API> \
    --output_name=LongForm-Qwen3-Next-80B-A3B-Thinking \
    --output_dir=/capstor/scratch/cscs/"${USER}"/synthetic/ \
    --concurrency=192 \
    --max_tokens=1536 \
    --temperature=0.6 \
    --top_p=1.0 \
    --seed=42
"

# Submits the CMD to all the processes on all the nodes.
srun bash -c "$CMD"

echo "[sbatch-master] task finished"