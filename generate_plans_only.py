import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

import config
from pipeline import append_jsonl, ensure_dirs, load_seeds, setup_logging, call_chat
from prompts import (
    PLANNER_SYSTEM,
    build_planner_user,
    load_fewshot_text,
    load_planner_template_text,
    load_taxonomy_text,
)


async def generate_plan_only(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    seed: Dict[str, Any],
    plan_idx: int,
    run_id: str,
    taxonomy_text: str,
    fewshot_text: str,
    planner_template: str,
) -> Dict[str, Any]:
    user_msg = build_planner_user(
        seed["user"],
        seed["assistant"],
        config.PLAN_TURNS,
        taxonomy_text,
        fewshot_text,
        planner_template,
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
        max_tokens=config.PLANNER_MAX_TOKENS,
        request_type="planner",
    )
    return {
        "seed": seed,
        "plan_idx": plan_idx,
        "plan_text": content,
        "run_id": run_id,
        "model": config.PLANNER_MODEL,
    }


async def main() -> None:
    run_id = datetime.utcnow().strftime("plans_only_%Y%m%d_%H%M%S")
    setup_logging(run_id)
    ensure_dirs()
    api_key = os.environ.get(config.API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing API key env var {config.API_KEY_ENV}")
    client = AsyncOpenAI(base_url=config.BASE_URL, api_key=api_key)
    sem = asyncio.Semaphore(config.MAX_CONCURRENCY)
    taxonomy_text = load_taxonomy_text() if config.USE_TAXONOMY else ""
    fewshot_text = load_fewshot_text() if config.USE_FEWSHOT else ""
    planner_template = load_planner_template_text() if config.USE_AGENT_TEMPLATE else ""
    seeds = load_seeds(30)
    out_path = config.PLANS_DIR / f"{run_id}.jsonl"
    logging.info("Writing planner-only outputs to %s", out_path)

    for seed in seeds:
        tasks: List[asyncio.Task] = []
        for plan_idx in range(config.PLANS_PER_SEED):
            tasks.append(
                asyncio.create_task(
                    generate_plan_only(
                        client,
                        sem,
                        seed,
                        plan_idx,
                        run_id,
                        taxonomy_text,
                        fewshot_text,
                        planner_template,
                    )
                )
            )
        plans = await asyncio.gather(*tasks)
        for plan in plans:
            append_jsonl(out_path, plan)


if __name__ == "__main__":
    asyncio.run(main())
