"""
Disk-backed JSON cache for FinBERT predictions -- GeoQuant AI Bloc 2.

Original author : Arthur (branch news_Arthur).
Integrated into NewsElena branch by Elena.

Avoids re-running FinBERT on texts that were already scored in a
previous session. The cache is stored in data/processed/sentiment_cache.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# Default location: inside the project, not at repo root
_DEFAULT_CACHE = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "sentiment_cache.json"
)


def load_cache(path: Path = _DEFAULT_CACHE) -> Dict[str, Any]:
    """
    Load the sentiment cache from disk.

    Args:
        path: Path to the JSON cache file.

    Returns:
        Dictionary mapping text -> prediction dict.
        Returns an empty dict if the file does not exist yet.
    """
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Any], path: Path = _DEFAULT_CACHE) -> None:
    """
    Persist the sentiment cache to disk.

    Args:
        cache: Dictionary to persist.
        path:  Target JSON file path (parent dirs are created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
