import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import psutil


def file_size_mb(path: Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 ** 2)


def directory_size_mb(path: Path) -> float:
    path = Path(path)
    total = 0
    if not path.exists():
        return 0.0
    for root, _, files in os.walk(path):
        for f in files:
            total += (Path(root) / f).stat().st_size
    return total / (1024 ** 2)


def current_process_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
