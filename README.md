# Multi-IF evaluation
multi-turn evaluation benchmark from https://github.com/facebookresearch/Multi-IF

## Files

* `api_client.py`: This file contains the implementation of the interface for LLMs interactions via API calls.
* `ifeval.py`: This file contains the implementation of the Inference Evaluation (IFEVAL) metric, which is used to evaluate the capability of LLM following natural language instructions
* `metrics.py`: This file contains the implementation of various metrics that can be used to calculate ifeval, data preprocess and enrichment for multi turn instructions.
* `utils.py`: This file contains utility functions that are used throughout the framework
* `multi_turn_instruct_following_eval_api.py`: This file contains the main function that executes the multi-turn evaluation benchmark via API calls.
* `multi_turn_instruct_following_eval_api_async.py`: This file contains the main function that executes the multi-turn evaluation benchmark via API calls, concurrency is based on `asyncio`.

## Usage
1. Create a new env with python 3.10 and install the required dependencies:
```bash
pip install -r requirements.txt
```
2. Download the data from huggingface:
```
git clone https://huggingface.co/datasets/facebook/Multi-IF data/Multi-IF
```

3. Download punkt_tab in python
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

4. Run experiments via `sbatch` file provided in the `experiments` branch.

## Results
Our final results corresponding to `Qwen3-8B-sft-mco-scaled` (use epoch 1 ckpt of the total 2 epochs) locate in `exp7_num_ep/all_1ep/Qwen3-8B-sft-mco-scaled`.