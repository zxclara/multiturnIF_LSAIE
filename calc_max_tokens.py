import json
from transformers import AutoTokenizer

def get_max_num_tokens(texts: list[str] | str):
    if isinstance(texts, str):
        texts = [ texts ]
    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5-Air-FP8")
    enc = tokenizer(texts, add_special_tokens=False)
    return max([len(ids) for ids in enc["input_ids"]])

path = "./generated/dialogues/run_20251209_001012.jsonl"

total_msgs = [ ]
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)
            total_msg = ''.join([ msg['content'] for msg in obj["messages"] ])
            total_msgs.append(total_msg)
print(f"max # tokens: {get_max_num_tokens(total_msgs)}")
