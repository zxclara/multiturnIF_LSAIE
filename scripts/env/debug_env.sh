#!/bin/bash
srun --account=infra01 --environment=/iopsstor/scratch/cscs/$USER/project/env/debug/nvidia_env.toml -p debug --pty bash