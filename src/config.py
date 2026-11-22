"""Config loading, paths and the global random seed.

Everything tunable lives in configs/config.yaml. Call load_config() rather than
hard-coding thresholds; call rng() rather than seeding numpy yourself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed() -> int:
    return int(load_config()["seed"])


def rng(offset: int = 0) -> np.random.Generator:
    """Generator seeded from the global seed (offset lets independent streams stay reproducible)."""
    return np.random.default_rng(seed() + offset)


@dataclass(frozen=True)
class Paths:
    raw: Path
    processed: Path
    tables: Path
    figures: Path

    @classmethod
    def from_config(cls, cfg: dict | None = None, root: Path | str = ROOT, out_root: Path | str | None = None) -> "Paths":
        """`out_root` redirects processed/tables/figures (used by tests); raw always reads from `root`."""
        p = (cfg or load_config())["paths"]
        root, out = Path(root), Path(out_root) if out_root else Path(root)
        return cls(raw=root / p["raw"], processed=out / p["processed"], tables=out / p["tables"], figures=out / p["figures"])

    def ensure(self) -> "Paths":
        for d in (self.processed, self.tables, self.figures):
            d.mkdir(parents=True, exist_ok=True)
        return self
