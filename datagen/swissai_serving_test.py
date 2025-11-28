import argparse
from typing import List, Dict

import openai

from utils.timer_utils import measure_throughput

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Next-80B-A3B-Thinking", help="Assistant model")
    parser.add_argument("--swissai_api", type=str, required=True, help="API for SwissAPI Serving Platform")
    parser.add_argument("--max_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser

def query(
    client: openai.Client,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> List[openai.types.chat.chat_completion_chunk.ChatCompletionChunk]:
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    for chunk in res:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    return res

def main(args: dict):
    api_key = args.pop("swissai_api")
    model = args.pop("model")
    client = openai.Client(api_key=api_key, base_url="https://api.swissai.cscs.ch/v1")
    
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    seed = args.pop("seed")
    
    num_instances = 1

    @measure_throughput(num_instances=num_instances)
    def generate_response():
        messages = [
            {
                "content": "Who is Pablo Picasso?", 
                "role": "user",
            }
        ]
        results = []
        for _ in range(num_instances):
            res = query(client, model, messages, max_tokens, temperature, top_p, seed)
            results.append(res)
        return results
    return generate_response()


if __name__=='__main__':
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
