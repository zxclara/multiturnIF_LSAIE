import asyncio
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

import config
from pipeline import append_jsonl, ensure_dirs, load_seeds, setup_logging, call_chat
from prompts import (
    CATEGORY_ORDER,
    build_planner_user,
    build_planner_system,
    category_to_axis,
    load_fewshot_text,
    load_planner_template_text,
    load_taxonomy_categories,
)


async def generate_plan_only(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    seed: Dict[str, Any],
    plan_idx: int,
    run_id: str,
    category_name: str,
    category_details: str,
    fewshot_text: str,
    planner_template: str,
    planner_system: str,
) -> Dict[str, Any]:
    switch_hint = random.random() < 0.2
    user_msg = build_planner_user(
        seed["user"],
        seed["assistant"],
        config.PLAN_TURNS,
        category_name,
        category_details,
        fewshot_text,
        planner_template,
        switch_hint,
    )
    messages = [
        {"role": "system", "content": planner_system},
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
        "twist": switch_hint,
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
    taxonomy_categories = load_taxonomy_categories()
    planner_template = load_planner_template_text() if config.USE_AGENT_TEMPLATE else ""
    seeds = load_seeds(30)
    out_path = config.PLANS_DIR / f"{run_id}.jsonl"
    logging.info("Writing planner-only outputs to %s", out_path)

    category_order = CATEGORY_ORDER if CATEGORY_ORDER else ["Instruction Retention"]

    for seed in seeds:
        tasks: List[asyncio.Task] = []
        for plan_idx in range(config.PLANS_PER_SEED):
            category_name = category_order[plan_idx % len(category_order)]
            category_details = taxonomy_categories.get(category_name, "")
            axis = category_to_axis(category_name)
            fewshot_text = load_fewshot_text(axis=axis) if config.USE_FEWSHOT else ""
            planner_system = build_planner_system(category_name)
            tasks.append(
                asyncio.create_task(
                    generate_plan_only(
                        client,
                        sem,
                        seed,
                        plan_idx,
                        run_id,
                        category_name,
                        category_details,
                        fewshot_text,
                        planner_template,
                        planner_system,
                    )
                )
            )
        plans = await asyncio.gather(*tasks)
        for plan in plans:
            append_jsonl(out_path, plan)


if __name__ == "__main__":
    asyncio.run(main())
