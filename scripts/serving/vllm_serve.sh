#!/bin/bash
vllm serve --config /iopsstor/scratch/cscs/${USER}/project/configs/serving/olmo3_7b_instruct.yaml &
vllm serve --config /iopsstor/scratch/cscs/"${USER}"/project/configs/serving/qwen3_next_80b_instruct.yaml &