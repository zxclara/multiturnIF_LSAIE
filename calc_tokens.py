import json
from transformers import AutoTokenizer

def get_num_tokens(texts: list[str] | str):
    if isinstance(texts, str):
        texts = [ texts ]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
    enc = tokenizer(texts, add_special_tokens=False)
    print(f"avg # tokens: {sum([len(ids) for ids in enc["input_ids"]]) / len(texts)}")
    print(f"max # tokens: {max([len(ids) for ids in enc["input_ids"]])}")

path = "./generated/dialogues/filtered/generated_mc_data.jsonl"

total_msgs = [ ]
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)
            total_msg = ''.join([ msg['content'] for msg in obj["messages"] ])
            total_msgs.append(total_msg)
get_num_tokens(total_msgs)