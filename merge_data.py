import json
import glob
from typing import Iterable, Dict, Any


def read_jsonl(file_path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a jsonl file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {file_path} at line {line_num}") from e


def filter_record(record: Dict[str, Any]) -> bool:
    if "messages" not in record:
        return False
    if record["messages"] == []:
        return False
    if record.get("selected", 0) == False:
        return False
    return True


def merge_and_filter(
    input_pattern: str,
    output_file: str
) -> None:
    """
    Read jsonl files matching input_pattern,
    merge and filter records,
    and write to output_file.
    """
    input_files = glob.glob(input_pattern)

    if not input_files:
        raise FileNotFoundError(f"No files match pattern: {input_pattern}")

    total_read = 0
    total_written = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for file_path in input_files:
            for record in read_jsonl(file_path):
                total_read += 1
                if filter_record(record):
                    # simple
                    record = {"messages": record["messages"], "selected": record["selected"]}
                    # total: no above assignment
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_written += 1

    print(f"Read {total_read} records")
    print(f"Wrote {total_written} filtered records to {output_file}")


if __name__ == "__main__":
    # Example usage
    merge_and_filter(
        input_pattern="generated/dialogues/*.jsonl",
        output_file="generated/dialogues/filtered/generated_mc_data.jsonl"
    )
