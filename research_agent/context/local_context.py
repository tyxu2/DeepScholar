from __future__ import annotations

import os
from pathlib import Path

_ALLOWED_EXTS = {
    ".tex", ".bib", ".sty", ".cls",
    ".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c", ".h",
    ".md", ".txt", ".rst",
    ".json", ".yaml", ".yml", ".toml", ".ini",
}


def resolve_input_paths(paths: list[str] | None, max_files: int = 24) -> list[str]:
    """展开文件/目录路径，返回可读文本文件绝对路径列表。"""
    if not paths:
        return []

    resolved: list[str] = []
    seen: set[str] = set()

    for raw in paths:
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix.lower() in _ALLOWED_EXTS:
                ap = str(p)
                if ap not in seen:
                    seen.add(ap)
                    resolved.append(ap)
            continue

        # directory
        for root, _, files in os.walk(str(p)):
            for name in files:
                fp = os.path.join(root, name)
                ext = Path(fp).suffix.lower()
                if ext not in _ALLOWED_EXTS:
                    continue
                ap = str(Path(fp).resolve())
                if ap in seen:
                    continue
                seen.add(ap)
                resolved.append(ap)
                if len(resolved) >= max_files:
                    return resolved
    return resolved[:max_files]


def _read_text(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def build_local_context(
    paths: list[str] | None,
    max_files: int = 12,
    per_file_chars: int = 1800,
    total_chars: int = 10000,
) -> tuple[list[str], str]:
    """
    构建本地文件上下文摘要。
    返回 (resolved_paths, context_text)。
    """
    files = resolve_input_paths(paths, max_files=max_files)
    if not files:
        return [], ""

    blocks: list[str] = []
    used = 0
    for fp in files:
        rel = os.path.relpath(fp, os.getcwd())
        content = _read_text(fp, per_file_chars)
        if not content:
            continue
        block = f"[FILE] {rel}\n{content.strip()}\n"
        if used + len(block) > total_chars:
            remain = max(total_chars - used, 0)
            if remain > 240:
                blocks.append(block[:remain])
            break
        blocks.append(block)
        used += len(block)

    header = (
        "以下是用户提供的本地文件片段（代码/模板/草稿），"
        "写作或规划时应优先遵循其术语、结构和约束：\n\n"
    )
    return files, header + "\n---\n".join(blocks)
