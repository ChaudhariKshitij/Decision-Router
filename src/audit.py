"""Append-only JSONL audit log. Records contain model outputs and routing metadata only (no identifiers/PII)."""
from __future__ import annotations

import json
import os
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from .config import artifacts_dir


def new_request_id() -> str:
    return uuid.uuid4().hex


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AuditLogger:
    """Thread-safe JSONL writer with a small in-memory index for O(1) recent lookups."""

    def __init__(self, path: str | os.PathLike | None = None, cache_size: int = 5000) -> None:
        self.path = Path(path or os.getenv("DR_AUDIT_PATH", artifacts_dir() / "audit" / "decisions.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = cache_size

    def log(self, record: dict) -> None:
        line = json.dumps(record, default=_json_default)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._cache[record["request_id"]] = record
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def get(self, request_id: str) -> dict | None:
        with self._lock:
            if request_id in self._cache:
                return self._cache[request_id]
        if not self.path.exists():
            return None
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if request_id in line:
                    rec = json.loads(line)
                    if rec.get("request_id") == request_id:
                        return rec
        return None


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:  # pragma: no cover
        pass
    return str(o)
