import json
import tiktoken

def get_num_tokens(text: str):
    encoding = tiktoken.get_encoding("o200k_harmony")
    tokens = encoding.encode(text)
    return len(tokens)

path = "./generated/dialogues/run_20251208_092349.jsonl"

max_cnt = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)
            total_msg = '\n\n'.join([ msg['content'] for msg in obj["messages"] ])
            max_cnt = max(max_cnt, get_num_tokens(total_msg))
print(f"max # tokens: {max_cnt}")
