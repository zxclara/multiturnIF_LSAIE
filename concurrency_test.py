import argparse
import asyncio
import os
import time
from typing import List, Tuple

from openai import AsyncOpenAI

import config


async def ping(client: AsyncOpenAI, idx: int) -> Tuple[float, str]:
    start = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=config.RESPONDER_MODEL,
            messages=[{"role": "user", "content": f"ping {idx}"}],
            temperature=0.0,
        )
        # print(resp)
        _ = resp.choices[0].message.content
        return time.perf_counter() - start, ""
    except Exception as e:
        return time.perf_counter() - start, str(e)


async def run_burst(concurrency: int, requests: int) -> None:
    api_key = os.environ.get(config.API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing API key env var {config.API_KEY_ENV}")
    client = AsyncOpenAI(base_url=config.BASE_URL, api_key=api_key)
    tasks: List[asyncio.Task] = []
    sem = asyncio.Semaphore(concurrency)

    async def guarded(idx: int) -> Tuple[float, str]:
        async with sem:
            return await ping(client, idx)

    start = time.perf_counter()
    for i in range(requests):
        tasks.append(asyncio.create_task(guarded(i)))
    results = await asyncio.gather(*tasks, return_exceptions=False)
    total = time.perf_counter() - start

    durations = [r[0] for r in results]
    errors = [r[1] for r in results if r[1]]
    successes = requests - len(errors)
    avg = sum(durations) / len(durations) if durations else 0

    print(
        f"Requests={requests}, concurrency={concurrency}, "
        f"avg_latency={avg:.2f}s, wall={total:.2f}s, "
        f"success={successes}, errors={len(errors)}"
    )
    if errors:
        unique = []
        seen = set()
        for err in errors:
            if err not in seen:
                unique.append(err)
                seen.add(err)
            if len(unique) >= 3:
                break
        print("Sample errors (up to 3 unique):")
        for i, err in enumerate(unique, 1):
            print(f"{i}. {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SwissAI concurrency.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=32)
    args = parser.parse_args()
    asyncio.run(run_burst(args.concurrency, args.requests))


if __name__ == "__main__":
    main()
