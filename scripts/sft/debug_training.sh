#!/bin/bash
tune run --nnodes 1 --nproc_per_node 4 sft_full_distributed --config /iopsstor/scratch/cscs/$USER/project/configs/sft/qwen3/8B_full_if.yaml