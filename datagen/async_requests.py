import json
import uuid
import asyncio
import pathlib
import argparse
from typing import List, Dict

import openai
import datasets
import pyarrow
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm_asyncio

from utils.timer_utils import measure_throughput

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Next-80B-A3B-Thinking", help="Assistant model")
    parser.add_argument("--swissai_api", type=str, required=True, help="API for SwissAPI Serving Platform")
    parser.add_argument("--max_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--output_name", type=str, default="tempdata")
    parser.add_argument("--output_dir", type=str, default="/capstor/scratch/cscs/yanghao/synthetic/")
    return parser

def get_dataset(source: str):
    if source == "": # dummy input
        dataset = datasets.load_dataset("akoksal/LongForm", cache_dir="/capstor/scratch/cscs/yanghao/datasets/")['test'][:512]
        dataset = [
            [ {'content': instance, 'role': "user"}, ]
            for instance in dataset['input']   
        ]
    else:
        dataset = datasets.load_dataset(source)
        # pre-processing here
    return dataset

async def run_single_turn_conversation(
    client: openai.AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    semaphore: asyncio.Semaphore,
) -> List[Dict[str, str]]:
    async with semaphore:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        reply = []
        async for chunk in stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                reply.append(chunk.choices[0].delta.content)
        reply = "".join(reply)
        messages.append({"role": "assistant", "content": reply})
        return messages

@measure_throughput(type="I/O", num_instances=1)
def save_synthetic_data(name, metadata, dataset, synthetic_dataset, target_path: str):
    num_instances = len(synthetic_dataset)
    data_dict = {
        "id": [ name + str(uuid.uuid1()) for _ in range(num_instances) ],
        "prompt": [ messages[0]['content'] for messages in dataset ],
        "messages": [ messages for messages in synthetic_dataset ],
        "contraints": [ {"dummy_class": "dummy_constraint"} for _ in range(num_instances) ],
    }
    data_table = pyarrow.Table.from_pydict(data_dict)
    data_path = pathlib.Path(target_path).joinpath(name)
    if not data_path.exists():
        data_path.mkdir()
        data_path.joinpath("data").mkdir()
    else:
        raise FileExistsError(f"[ERROR] data path: {str(data_path)} exists...")
    pq.write_table(data_table, data_path.joinpath("data/data.parquet"))
    with open(data_path.joinpath("metadata.json"), 'w') as fp:
        json.dump(metadata, fp)
    print(f"finish saving synthetic data to {str(data_path)}...")

def main(args: dict):
    api_key = args.pop("swissai_api")
    model = args.pop("model")
    client = openai.AsyncOpenAI(api_key=api_key, base_url="https://api.swissai.cscs.ch/v1")
    
    concurrency = args.pop("concurrency")
    semaphore = asyncio.Semaphore(concurrency)

    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    seed = args.pop("seed")
    
    source = args.pop("source")
    dataset = get_dataset(source)
    num_instances = len(dataset)

    @measure_throughput(type="inference", num_instances=num_instances)
    def generate_synthetic_dataset(dataset: List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
        async def generate_responses(messages_list: List[List[Dict[str, str]]]):
            tasks = [ 
                asyncio.create_task(
                    run_single_turn_conversation(
                        client, model, messages, 
                        max_tokens, temperature, top_p, seed,
                        semaphore,
                    )
                ) for messages in messages_list 
            ]
            # collect results
            results = await tqdm_asyncio.gather(*tasks, desc="Generating conversations")
            # results = await asyncio.gather(*tasks)
            return results
        
        results = asyncio.run(generate_responses(dataset))
        return results
    synthetic_dataset = generate_synthetic_dataset(dataset)

    metadata = {
        "seed_prompt": None,
        "prompt_template": None,
        "sampling_params": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        },
        "model": model,
        "turns": len(synthetic_dataset[0]) if synthetic_dataset else None,
    }
    print(f"metadata: {metadata}\nsynthetic_dataset example: {synthetic_dataset[0]}")
    if len(synthetic_dataset) >= 2:
        print(f"answer 1 == answer 2? {synthetic_dataset[0] == synthetic_dataset[1]}")
        print(f"first 256 characters: ")
        print(f"answer 1: {synthetic_dataset[0][-1]['content'][:256]}\n")
        print(f"answer 2: {synthetic_dataset[1][-1]['content'][:256]}")

    synthetic_data_name = args.pop("output_name")
    target_path = args.pop("output_dir")
    save_synthetic_data(synthetic_data_name, metadata, dataset, synthetic_dataset, target_path)

if __name__=='__main__':
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
