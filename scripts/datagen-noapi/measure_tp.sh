#!/bin/bash
# LongForm measurement (test split: 512 instances)
## Qwen3-32B, scrachpad
python async_requests.py --model=Qwen/Qwen3-32B --swissai_api=<YOUR_API> --output_name=LongForm-Qwen3-32B --concurrency=128
## Qwen3-Next-80B-A3B-Thinking, SwissAI Platform
python async_requests.py --model=Qwen/Qwen3-Next-80B-A3B-Thinking --swissai_api=<YOUR_API> --output_name=LongForm-Qwen3-Next-80B-A3B-Thinking --concurrency=192
