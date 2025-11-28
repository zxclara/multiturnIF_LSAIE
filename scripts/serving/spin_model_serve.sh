#!/bin/bash
conda deactivate # run this script with `source xxx.sh`
# Qwen/Qwen3-30B-A3B-Instruct-2507
spin-model --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp-size 4 --time 1h --account infra01 