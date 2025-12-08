import asyncio
import json
from pathlib import Path
from typing import Any, Dict

class AsyncJSONLWriter:
    def __init__(self, path: Path):
        self.path = path
        self.queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start the writer task"""
        self._task = asyncio.create_task(self._writer())

    async def _writer(self):
        with self.path.open("a", encoding="utf-8") as f:
            while True:
                item = await self.queue.get()
                if item is None:
                    break
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                self.queue.task_done()

    async def write(self, data: Dict[str, Any]):
        """Enqueue data for writing"""
        await self.queue.put(data)

    async def stop(self):
        """Stop the writer and flush remaining items"""
        await self.queue.join()
        await self.queue.put(None)
        if self._task:
            await self._task
