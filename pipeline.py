import asyncio
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from openai import AsyncOpenAI

import config
from prompts import (
    PLANNER_SYSTEM,
    build_evaluator_prompt,
    build_planner_user,
    build_responder_system,
    load_fewshot_text,
    load_taxonomy_text,
)


def ensure_dirs() -> None:
    for path in [
        config.HF_CACHE,
        config.PLANS_DIR,
        config.DIALOGUES_DIR,
        config.LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(run_id: str) -> None:
    ensure_dirs()
    log_path = config.LOG_DIR / f"{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def parse_json_maybe(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        logging.warning("Failed to parse JSON; returning empty dict. Raw: %s", text)
        return {}


def extract_seed(record: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    # Try common conversation layouts
    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list) and len(messages) == 2:
        user_msg = messages[0]
        assistant_msg = messages[1]
        if (
            isinstance(user_msg, dict)
            and isinstance(assistant_msg, dict)
            and user_msg.get("role") == "user"
            and assistant_msg.get("role") == "assistant"
        ):
            return {
                "source_id": record.get("id", idx),
                "user": user_msg.get("content", ""),
                "assistant": assistant_msg.get("content", ""),
            }
    # Try instruction/output pairs
    if "instruction" in record and "output" in record:
        return {
            "source_id": record.get("id", idx),
            "user": record.get("instruction", ""),
            "assistant": record.get("output", ""),
        }
    return None


def load_seeds(limit: int) -> List[Dict[str, Any]]:
    ds = load_dataset(
        config.DATASET_ID,
        split="train",
        cache_dir=str(config.HF_CACHE),
    )
    seeds = []
    for idx, rec in enumerate(ds):
        seed = extract_seed(rec, idx)
        if seed:
            seeds.append(seed)
        if len(seeds) >= limit:
            break
    logging.info("Loaded %d seeds (limit=%d)", len(seeds), limit)
    return seeds


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


async def call_chat(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    timeout: int,
    max_tokens: Optional[int] = None,
    request_type: Optional[str] = None,
) -> str:
    # Log the first payload per request type to help debugging.
    if not hasattr(call_chat, "_logged_types"):
        call_chat._logged_types = set()  # type: ignore[attr-defined]
    if request_type and request_type not in call_chat._logged_types:  # type: ignore[attr-defined]
        try:
            logging.info(
                "First %s request payload: %s",
                request_type,
                json.dumps(
                    {
                        "model": model,
                        "temperature": temperature,
                        "timeout": timeout,
                        "max_tokens": max_tokens,
                        "messages": messages,
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception:
            logging.info("First %s request payload logging failed", request_type)
        call_chat._logged_types.add(request_type)  # type: ignore[attr-defined]

    last_err: Optional[Exception] = None
    for attempt in range(config.MAX_RETRIES):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                    max_tokens=max_tokens,
                )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt * 0.5, 8.0) + random.uniform(0, 0.25)
            logging.warning(
                "chat error (attempt %d/%d): %s; retrying in %.2fs; model=%s",
                attempt + 1,
                config.MAX_RETRIES,
                e,
                wait,
                model,
            )
            await asyncio.sleep(wait)
    logging.error("chat failed after %d attempts; last error: %s", config.MAX_RETRIES, last_err)
    raise last_err if last_err else RuntimeError("Unknown chat error")


def compute_score(evaluation: Dict[str, Any]) -> float:
    def to_num(val: Any) -> float:
        try:
            return float(val)
        except Exception:
            return 0.0

    return sum(
        [
            to_num(evaluation.get("naturalness", 0)),
            to_num(evaluation.get("authenticity", 0)),
            to_num(evaluation.get("safety", 0)),
            to_num(evaluation.get("accuracy", 0)),
        ]
    )


async def generate_plan(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    seed: Dict[str, Any],
    plan_idx: int,
    run_id: str,
    taxonomy_text: str,
    fewshot_text: str,
) -> Dict[str, Any]:
    user_msg = build_planner_user(
        seed["user"],
        seed["assistant"],
        config.PLAN_TURNS,
        taxonomy_text,
        fewshot_text,
    )
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    content = await call_chat(
        client,
        sem,
        model=config.PLANNER_MODEL,
        messages=messages,
        temperature=config.PLANNER_TEMPERATURE,
        timeout=config.DEFAULT_TIMEOUT,
        max_tokens=config.MAX_TOKENS,
        request_type="planner",
    )
    parsed = parse_json_maybe(content)
    parsed.setdefault("questions", [])
    parsed.setdefault("trap_summary", "")
    parsed.setdefault("detection", "")
    parsed.setdefault("challenge_type", "")
    parsed["raw"] = content
    parsed["plan_idx"] = plan_idx
    parsed["run_id"] = run_id
    return parsed


async def run_responder(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    seed: Dict[str, Any],
    plan: Dict[str, Any],
    traj_idx: int,
) -> List[Dict[str, str]]:
    system_msg = build_responder_system(seed["assistant"])
    history: List[Dict[str, str]] = [{"role": "system", "content": system_msg}]
    dialogue: List[Dict[str, str]] = []
    for q in plan.get("questions", [])[: config.PLAN_TURNS]:
        history.append({"role": "user", "content": q})
        answer = (
            await call_chat(
                client,
                sem,
                model=config.RESPONDER_MODEL,
                messages=history,
                temperature=config.RESPONDER_TEMPERATURE,
                timeout=config.DEFAULT_TIMEOUT,
                max_tokens=config.MAX_TOKENS,
                request_type="responder",
            )
        ).strip()
        history.append({"role": "assistant", "content": answer})
        dialogue.append({"role": "user", "content": q})
        dialogue.append({"role": "assistant", "content": answer})
    logging.info(
        "Built trajectory with %d messages for source_id=%s plan=%d traj=%d",
        len(dialogue),
        seed["source_id"],
        plan.get("plan_idx", -1),
        traj_idx,
    )
    return dialogue


async def evaluate_trajectory(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    plan: Dict[str, Any],
    dialogue: List[Dict[str, str]],
) -> Dict[str, Any]:
    plan_text = plan.get("raw", "")
    trap_summary = plan.get("trap_summary", "")
    detection = plan.get("detection", "")
    prompt = build_evaluator_prompt(plan_text, trap_summary, detection, dialogue)
    messages = [{"role": "user", "content": prompt}]
    content = await call_chat(
        client,
        sem,
        model=config.EVALUATOR_MODEL,
        messages=messages,
        temperature=config.EVALUATOR_TEMPERATURE,
        timeout=config.DEFAULT_TIMEOUT,
        max_tokens=config.MAX_TOKENS,
        request_type="evaluator",
    )
    parsed = parse_json_maybe(content)
    parsed["raw"] = content
    return parsed


def build_entry(
    seed: Dict[str, Any],
    plan: Dict[str, Any],
    dialogue: List[Dict[str, str]],
    evaluation: Dict[str, Any],
    plan_idx: int,
    traj_idx: int,
    run_id: str,
    selected: bool,
    score: float,
) -> Dict[str, Any]:
    entry_id = f"{seed['source_id']}_{plan_idx}_{traj_idx}"
    return {
        "id": entry_id,
        "source_id": seed["source_id"],
        "plan_idx": plan_idx,
        "traj_idx": traj_idx,
        "run_id": run_id,
        "planner_model": config.PLANNER_MODEL,
        "responder_model": config.RESPONDER_MODEL,
        "evaluator_model": config.EVALUATOR_MODEL,
        "challenge_type": plan.get("challenge_type", ""),
        "trap_summary": plan.get("trap_summary", ""),
        "plan_detection": plan.get("detection", ""),
        "plan_raw": plan.get("raw", ""),
        "messages": dialogue,
        "seed": {"user": seed["user"], "assistant": seed["assistant"]},
        "evaluation": evaluation,
        "score": score,
        "trap_present": evaluation.get("trap_present", False),
        "trap_triggered": evaluation.get("trap_triggered", False),
        "selected": selected,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


async def process_seed(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    seed: Dict[str, Any],
    run_id: str,
    plans_path: Path,
    dialogues_path: Path,
    taxonomy_text: str,
    fewshot_text: str,
) -> None:
    for plan_idx in range(config.PLANS_PER_SEED):
        plan = await generate_plan(
            client, sem, seed, plan_idx, run_id, taxonomy_text, fewshot_text
        )
        append_jsonl(plans_path, {"seed": seed, "plan": plan})
        entries = []

        async def run_one(traj_idx: int) -> Dict[str, Any]:
            try:
                dialogue = await run_responder(client, sem, seed, plan, traj_idx)
                evaluation = await evaluate_trajectory(client, sem, plan, dialogue)
                score = compute_score(evaluation)
                return {
                    "traj_idx": traj_idx,
                    "dialogue": dialogue,
                    "evaluation": evaluation,
                    "score": score,
                    "error": None,
                }
            except Exception as e:
                logging.error(
                    "Trajectory failed for source_id=%s plan=%d traj=%d: %s",
                    seed["source_id"],
                    plan_idx,
                    traj_idx,
                    e,
                )
                return {
                    "traj_idx": traj_idx,
                    "dialogue": [],
                    "evaluation": {"error": str(e)},
                    "score": 0.0,
                    "error": str(e),
                }

        tasks = [asyncio.create_task(run_one(traj_idx)) for traj_idx in range(config.TRAJECTORIES_PER_PLAN)]
        entries = await asyncio.gather(*tasks)

        # Select top-2 among trap-present and trap-not-triggered
        eligible = [
            e
            for e in entries
            if not e.get("error")
            and e["evaluation"].get("trap_present")
            and not e["evaluation"].get("trap_triggered")
        ]
        eligible_sorted = sorted(eligible, key=lambda e: e["score"], reverse=True)
        keep_ids = {e["traj_idx"] for e in eligible_sorted[:2]}

        for item in entries:
            selected = item["traj_idx"] in keep_ids
            entry = build_entry(
                seed=seed,
                plan=plan,
                dialogue=item["dialogue"],
                evaluation=item["evaluation"],
                plan_idx=plan_idx,
                traj_idx=item["traj_idx"],
                run_id=run_id,
                selected=selected,
                score=item["score"],
            )
            append_jsonl(dialogues_path, entry)


async def main() -> None:
    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    setup_logging(run_id)
    ensure_dirs()
    api_key = os.environ.get(config.API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing API key env var {config.API_KEY_ENV}")
    client = AsyncOpenAI(base_url=config.BASE_URL, api_key=api_key)
    sem = asyncio.Semaphore(config.MAX_CONCURRENCY)
    taxonomy_text = load_taxonomy_text() if config.USE_TAXONOMY else ""
    fewshot_text = load_fewshot_text() if config.USE_FEWSHOT else ""
    seeds = load_seeds(config.SEEDS_PER_RUN)
    plans_path = config.PLANS_DIR / f"{run_id}.jsonl"
    dialogues_path = config.DIALOGUES_DIR / f"{run_id}.jsonl"
    for seed in seeds:
        await process_seed(
            client,
            sem,
            seed,
            run_id,
            plans_path,
            dialogues_path,
            taxonomy_text,
            fewshot_text,
        )


if __name__ == "__main__":
    asyncio.run(main())
