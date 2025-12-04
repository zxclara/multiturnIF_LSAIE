#!/bin/bash
conda deactivate # run this script with `source xxx.sh`
# Qwen/Qwen3-30B-A3B-Instruct-2507
spin-model --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp-size 4 --time 1h --account infra01 
# Qwen/Qwen3-Next-80B-A3B-Thinking
spin-model --model Qwen/Qwen3-Next-80B-A3B-Thinking --tp-size 4 --time 1h --account infra01 --vllm