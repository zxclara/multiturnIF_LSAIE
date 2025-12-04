#!/bin/bash
# LongForm measurement (test split: 512 instances)
## Qwen3-32B, scrachpad
python async_requests.py --model=Qwen/Qwen3-32B --swissai_api=<YOUR_API> --output_name=LongForm-Qwen3-32B --concurrency=128
## Qwen3-Next-80B-A3B-Thinking, SwissAI Platform
python async_requests.py --model=Qwen/Qwen3-Next-80B-A3B-Thinking --swissai_api=<YOUR_API> --output_name=LongForm-Qwen3-Next-80B-A3B-Thinking --concurrency=192
## Apertus-70B, SwissAI Platform
python async_requests.py --model=swiss-ai/Apertus-70B-Instruct-2509 --swissai_api=<YOUR_API> --output_name=LongForm-Apertus-70B-Instruct-2509 --concurrency=192
## Olmo-3-7B-Instruct, local vLLM
python async_requests.py --model=allenai/Olmo-3-7B-Instruct --swissai_api=none --base_url=http://172.28.36.112:8080/v1 --output_name=LongForm-Olmo-3-7B-Instruct --concurrency=128
## Qwen3-Next-80B-A3B-Instruct, local vLLM
python async_requests.py --model=Qwen/Qwen3-Next-80B-A3B-Instruct --swissai_api=none --base_url=http://172.28.42.88:8080/v1 --output_name=LongForm-Qwen3-Next-80B-A3B-Instruct --concurrency=128
## Olmo-3-32B-Think, local vLLM
python async_requests.py --model=allenai/Olmo-3-32B-Think --swissai_api=none --base_url=http://172.28.46.124:8080/v1 --output_name=LongForm-Olmo-3-32B-Think --concurrency=128
