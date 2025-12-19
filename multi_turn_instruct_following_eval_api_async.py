# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import time
import asyncio

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from api_client import get_api_bot
from metrics import MultiTurnInstructionFollowingPromptSolution
from utils import GenerationSetting

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


async def max_retry_wrapper_async(api_bot, messages, max_retry=3):
    for attempt in range(max_retry, 0, -1):
        try:
            response = await api_bot.async_generate(messages)
            return response
        except Exception as e:
            print(messages)
            logger.error(f"API call failed with error: {e}. Retries left: {attempt - 1}")
            await asyncio.sleep(1)  # Brief pause before retrying
    return f'[MAX_RETRY=0] Failed.'

async def async_process_row(api_bot, row, step, max_retry):
    try:
        if step == 1:
            messages = [json.loads(row['turn_1_prompt'])]
        else:
            messages = json.loads(row['turns']) + [
                {'role': 'assistant', 'content': row['responses']},
                json.loads(row[f'turn_{step}_prompt']),
            ]
        # use async wrapper instead of thread pool
        response = await max_retry_wrapper_async(api_bot, messages, max_retry)
        updated_turns = json.dumps(messages)
        status = 'success' if not response.startswith('[MAX_RETRY') else 'failed'
        return updated_turns, response, status
    except Exception as e:
        logger.exception(f"Error processing row: {e}")
        print(row)
        return row.get('turns', '[]'), f'Exception: {e}', 'exception'

async def process_row_with_semaphore(semaphore, api_bot, row, step, max_retry):
    async with semaphore:
        return await async_process_row(api_bot, row, step, max_retry)

async def process_rows_async_limited(input_df, output_df, api_bot, step, max_retry, max_concurrent=128) -> pd.DataFrame:
    # 1. Filter rows to process
    rows_to_process = []
    total_loc = len(input_df)
    for idx, row in input_df.iterrows():
        current_turn_index = row.get('turn_index', 0)
        response = row.get('responses', 'None')
        if current_turn_index > step or (current_turn_index == step and not response.startswith('[MAX_RETRY')):
            print(f"Skipped idx: {idx}")
            continue
        rows_to_process.append((idx, row))

    logger.info(f"Processing {len(rows_to_process)} out of {total_loc} rows for step {step}")

    # 2. Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    # 3. Create async tasks
    tasks = [
        process_row_with_semaphore(semaphore, api_bot, row, step, max_retry)
        for _, row in rows_to_process
    ]

    # 4. Run tasks concurrently with limited concurrency
    results_list = await tqdm_asyncio.gather(*tasks, desc="collect responses")

    # 5. Update output DataFrame
    for (idx, _), (turns, response, status) in zip(rows_to_process, results_list):
        output_df.at[idx, "turns"] = turns
        output_df.at[idx, "responses"] = response
        output_df.at[idx, "status"] = status

    return output_df

async def step_fn_api(
    api_bot,
    input_df,
    step,
    need_write2file=True,
    output_filepath=None,
    max_retry=3,
    max_workers=5,  # Limit the number of threads
):
    output_df = input_df.copy()
    if "turns" not in output_df.columns:
        output_df["turns"] = pd.array(["[]"] * len(output_df), dtype="string")
    if "responses" not in output_df.columns:
        output_df["responses"] = pd.array(["None"] * len(output_df), dtype="string")
    if "status" not in output_df.columns:
        output_df["status"] = pd.array(["pending"] * len(output_df), dtype="string")
    output_df['turn_index'] = step  # Update to current step

    output_df = await process_rows_async_limited(input_df, output_df, api_bot, step, max_retry, max_workers)

    if need_write2file and output_filepath:
        output_df.to_csv(output_filepath, index=False)
        logger.info(f"Step {step} results written to {output_filepath}")

    return output_df


def consolidate_results(api_model_name, output_filepath_prefix, steps):
    consolidated_df = None
    for step in steps:
        step_csv = f"results/{api_model_name}/{output_filepath_prefix}_step_{step}.csv"
        if os.path.exists(step_csv):
            temp_df = pd.read_csv(step_csv, keep_default_na=False)
            if consolidated_df is None:
                consolidated_df = temp_df.copy()
            else:
                # Merge on a unique identifier; assuming the index serves as a unique identifier
                consolidated_df = consolidated_df.combine_first(temp_df)
        else:
            logger.warning(f"Step {step} file {step_csv} does not exist and will be skipped.")

    if consolidated_df is not None:
        consolidated_csv = f"results/{api_model_name}/{output_filepath_prefix}_consolidated.csv"
        consolidated_df.to_csv(consolidated_csv, index=False)
        logger.info(f"All steps consolidated into {consolidated_csv}")
    else:
        logger.warning("No data available to consolidate.")


def main(
    api_model_name,
    input_data_csv: str = "dataset/multi_turn_sample.csv",
    generation_setting=GenerationSetting(
        max_new_tokens=32768, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, seed=42
    ),
    need_write2file: bool = True,
    output_filepath_prefix: str = "eval_result",
    max_workers: int = 5,  # Number of threads
    steps: list = [1, 2, 3],  # New parameter for steps
    vllm_ip: str = "",
):
    benchmark_df = pd.read_csv(input_data_csv, keep_default_na=False)
    num_rows = len(benchmark_df.axes[0])
    logger.info(f"Number of rows in input data: {num_rows}")
    final_metric_result = {}

    logger.info(f"generation_config: {generation_setting}")

    api_bot = get_api_bot(api_model_name, generation_setting, vllm_ip)
    step_input_df = benchmark_df.copy()
    for step in steps:  # Use the user-provided steps
        output_filepath = (
            f"results/{api_model_name}/{output_filepath_prefix}_step_{step}.csv"
        )
        os.makedirs(f'results/{api_model_name}', exist_ok=True)
        step_output_df = asyncio.run(step_fn_api(
            api_bot=api_bot,
            input_df=step_input_df,
            step=step,
            need_write2file=need_write2file,
            output_filepath=output_filepath,
            max_workers=max_workers,
        ))

        step_input_df = step_output_df.copy()
        step_metric_result = run_metric(
            api_model_name,
            output_filepath_prefix=output_filepath_prefix,
            step=step,
        )
        final_metric_result[step] = step_metric_result

    consolidate_results(api_model_name, output_filepath_prefix, steps=steps)


def run_metric(
    api_model_name,
    output_filepath_prefix: str = "eval_result",
    step: int = 1
):
    step_output_df = None
    step_csv = f"results/{api_model_name}/{output_filepath_prefix}_step_{step}.csv"
    if not os.path.exists(step_csv):
        logger.warning(f"CSV file {step_csv} does not exist and will be skipped.")
        return {}

    logger.info(f"Calculating metrics for step_{step}")
    step_output_df = pd.read_csv(step_csv, keep_default_na=False)

    if step_output_df is not None and not step_output_df.empty:
        metric_result = MultiTurnInstructionFollowingPromptSolution.metrics_gen(
            step_output_df
        )
        metric_result_df = pd.DataFrame.from_dict(metric_result, orient="index")
        metric_result_df.to_csv(f"results/{api_model_name}/{output_filepath_prefix}_step_{step}_metric.csv")
        logger.info(f"Step {step} metrics:\n{metric_result}")
        return metric_result
    else:
        logger.warning(f"No data available for step {step} to compute metrics.")
        return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_model_name", type=str, default="o1-mini")
    parser.add_argument(
        "--input_data_csv", type=str, default="dataset/multi_turn_sample.csv"
    )
    parser.add_argument("--need_write2file", type=bool, default=True)
    parser.add_argument("--output_filepath_prefix", type=str, default="eval_result")

    parser.add_argument('--max_new_tokens', type=int, default=32768, help='qwen3 uses 32768')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_workers', type=int, default=128, help='Number of async coros for concurrency')

    # New steps argument
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='List of steps to process (e.g., --steps 1 2 3)'
    )
    parser.add_argument('--vllm_ip', type=str, default="", help="vllm ip address for access")

    args = parser.parse_args()
    logger.info(f'Args: \n  max_new_tokens: {args.max_new_tokens}, \n  temperature: {args.temperature}, \n  top_p: {args.top_p}, \n  seed: {args.seed}, \n  max_workers: {args.max_workers}, \n  steps: {args.steps}')

    if 'o1' in args.api_model_name:
        # o1 doesn't allow for customized top_p and temperature.
        args.top_p = 1
        args.temperature = 1
    generation_setting = GenerationSetting(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed
    )

    main(
        api_model_name=args.api_model_name,
        input_data_csv=args.input_data_csv,
        generation_setting=generation_setting,
        need_write2file=args.need_write2file,
        output_filepath_prefix=args.output_filepath_prefix,
        max_workers=args.max_workers,
        steps=args.steps,
        vllm_ip=args.vllm_ip,
    )