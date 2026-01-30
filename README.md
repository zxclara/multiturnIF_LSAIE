# Modified Torchtune and Experiment Scripts

## 1. Modified Torchtune
We conducted supervised fine-tuning based on torchtune v0.6.1. Main changes included:
- top entry: `sft_full_distributed.py` modified from `recipe/full_finetune_distributed.py`
- IF dataset support: `torchtune/data/_messages.py` and `torchtune/datasets/_if.py`
- loss (avg/sum/scaled): `torchtune/modules/loss/cross_entropy_loss.py`

Torchtunes reads configuration `yaml` files in the `configs/sft/qwen3` folder. For the final reported experiment, we use `configs/sft/qwen3/8B_full_mco_scaled/ep2_lr1e-6.yaml`. For SFT evalutation, we used the `epoch 1` checkpoint of the total 2 epochs to avoid overfitting. Corresponding vLLM serving configuration file is `configs/serving/qwen3_8b_mco_scaled.yaml`.
- `mco`: multi-challenge-only, compared to the data mix of if-eval and multi-turn multi-challenge.
- `scaled`: scaled loss (dynamic SFT).

Specifically, we used a global batch size of 32, sequence length of 16,384 (w/ sample packing), on 8 Nvidia GH200 GPUs (generously provided by the Swiss AI Initiative) with ZeRO-2. And we run the SFT for 2 epochs with AdamW (default weight decay, lr=1e-6), and cosine annealing with warmup (warmup steps = 22).

By SFT with our generated `15104` multi-challenge English chats (max # tokens: 14894, avg # tokens: 3473, 3-8 turns), the final reported experiment on Multi-IF benchmark is:
|  | turn 1 | turn 2 | turn 3 |
| --- | --- | --- | --- |
| base (report) | N/A | N/A | 69.2 |
| base | 85.26 | 77.40 | 69.17 |
| mc, scaled loss | 85.84 | 77.60 | 69.03 |

A detailed observation on different languages showed that accuracy in `EN`, `FR`, `ZH` has improved a lot, compensating the decrease in other languages. This shows the necessity of multilingual multi-challenge synthetic data generation instead of current monolingual pattern.
|  | turn 1 | turn 1 | turn 1 | turn 2 | turn 2 | turn 2 | turn 3 | turn 3 | turn 3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | EN | FR | ZH | EN | FR | ZH | EN | FR | ZH |
| base | 87.08 | 86.22 | 82.62 | 81.54 | 80.23 | 73.65 | 74.09 | 70.66 | 68.45 |
| mc, scaled | 88.75 | 86.70 | 84.04 | 83.02 | 80.41 | 75.43 | 74.92 | 71.15 | 68.55 |

## 2. Experiment Scripts
Here we introduce the relavant scripts for `vLLM serving`, `SFT`, and `SFT Eval with Multi-IF`.
- `config`: i. `serving` contains `yaml` files for base model serving (for synthetic data generation, includes `glm4.5_air.yaml`, `gpt_oss_120b.yaml`, and `qwen3_next_80b_instruct.yaml`) and fine-tuned model evaluation; ii. `sft` contains `yaml` files for different fine-tuning experiments.
- `env`: `if_multi_sft.toml` is our environment for SFT and SFT evaluation, with corresponding `Dockerfile` in `scripts/env`; `vllm_env.toml` is a simple wrapper over the official vLLM v0.11.0 environment to fit in with the `Alps` supercomputer.
- `scripts/datagen-noapi`: `planner.sbatch`, `responder.sbatch` and `evaluator.sbatch` corresponds to the 3 roles in our data generation pipeline.
- `scripts/serving`: contains template `sbatch` files for single-node and multi-node (via `ray`) vLLM serving.
- `scripts/sft`: contains `sbatch` entry for SFT under different settings. `*_grid.sbatch` runs grid search via `#SBATCH --array=0-5%3`. Our final result is based on `sft_mco_scaled.sbatch`.
- `scripts/sft_eval`: contains cpu-only `sbatch` entry for SFT evaluation. Our final result is based on `mco_scaled.sbatch`.