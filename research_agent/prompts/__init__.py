from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from research_agent.prompts.defaults import DEFAULT_PROMPTS

PROMPT_OVERRIDE_ENV = "RESEARCH_AGENT_PROMPT_OVERRIDES"
DEFAULT_OVERRIDE_PATH = "./templates/prompt_overrides.json"


def _normalize_prompt_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        k = key.strip()
        if not k:
            continue
        normalized[k] = value
    return normalized


@lru_cache(maxsize=1)
def _load_prompt_overrides() -> dict[str, str]:
    path = os.getenv(PROMPT_OVERRIDE_ENV, "").strip() or DEFAULT_OVERRIDE_PATH
    if not path:
        return {}
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _normalize_prompt_map(data)


def clear_prompt_cache() -> None:
    _load_prompt_overrides.cache_clear()


def list_prompt_ids() -> list[str]:
    ids = set(DEFAULT_PROMPTS.keys())
    ids.update(_load_prompt_overrides().keys())
    return sorted(ids)


def get_prompt_template(prompt_id: str) -> str:
    key = str(prompt_id).strip()
    if not key:
        raise KeyError("prompt_id is empty")
    overrides = _load_prompt_overrides()
    if key in overrides:
        return overrides[key]
    if key in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[key]
    raise KeyError(f"unknown prompt id: {key}")


def render_prompt(prompt_id: str, **kwargs: object) -> str:
    template = get_prompt_template(prompt_id)
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing = e.args[0] if e.args else "unknown"
        raise KeyError(f"missing variable '{missing}' for prompt '{prompt_id}'") from e
