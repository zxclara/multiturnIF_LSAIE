# Multi-IF evaluation
multi-turn evaluation benchmark from https://github.com/facebookresearch/Multi-IF

## Files

* `api_client.py`: This file contains the implementation of the interface for LLMs interactions via API calls.
* `ifeval.py`: This file contains the implementation of the Inference Evaluation (IFEVAL) metric, which is used to evaluate the capability of LLM following natural language instructions
* `metrics.py`: This file contains the implementation of various metrics that can be used to calculate ifeval, data preprocess and enrichment for multi turn instructions.
* `utils.py`: This file contains utility functions that are used throughout the framework
* `multi_turn_instruct_following_eval_api.py`: This file contains the main function that executes the multi-turn evaluation benchmark via API calls.

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

4. Swiss AI api key: export MY_SWISSAI_API=<your_key>

5. Run the evaluation in `multi_turn_instruct_following_eval_api.py` with `swiss-ai/Apertus-8B-Instruct-2509`:
```bash
python multi_turn_instruct_following_eval_api.py \
        --max_workers 5 \
        --api_model_name swiss-ai/Apertus-8B-Instruct-2509 \
        --input_data_csv data/Multi-IF/multiIF_20241018.csv \
        --max_new_tokens 1024 \
        --steps 1 2 3
```