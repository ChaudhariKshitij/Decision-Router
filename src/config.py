"""Configuration loading and path helpers. All thresholds come from configs/routing.yaml."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TIERS: tuple[str, ...] = ("fast", "accurate", "deep")


def load_config(path: str | os.PathLike | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load the YAML config (``DR_CONFIG`` env var or configs/routing.yaml) and deep-merge overrides."""
    p = Path(path or os.getenv("DR_CONFIG", ROOT / "configs" / "routing.yaml"))
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_update(copy.deepcopy(cfg), overrides)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge ``upd`` into ``base`` (returns base)."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def artifacts_dir() -> Path:
    d = Path(os.getenv("DR_ARTIFACTS_DIR", ROOT / "artifacts"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    d = Path(os.getenv("DR_DATA_DIR", ROOT / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = artifacts_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir() -> Path:
    d = artifacts_dir() / "reports"
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d
