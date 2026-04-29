import json
import re


def _try_repair_truncated(text: str) -> dict | None:
    """
    Best-effort repair of a truncated JSON object.
    Strategy: find the last complete key-value pair, close all open
    brackets/braces, and re-parse.
    """
    # Trim to the last successfully parseable position
    # Walk backward from end, counting open braces/brackets
    # Simple approach: find the last '}' that closes the root object
    # If the text starts with '{', try progressively shorter suffixes.
    if not text.strip().startswith("{"):
        return None

    # Try closing unclosed strings and brackets
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape_next = False
    last_safe_pos = 0  # position after last complete key-value pair at root level

    chars = list(text)
    i = 0
    while i < len(chars):
        c = chars[i]
        if escape_next:
            escape_next = False
        elif c == "\\" and in_string:
            escape_next = True
        elif c == '"' and not escape_next:
            in_string = not in_string
        elif not in_string:
            if c == "{":
                depth_brace += 1
            elif c == "}":
                depth_brace -= 1
                if depth_brace == 0:
                    last_safe_pos = i + 1
                    break
            elif c == "[":
                depth_bracket += 1
            elif c == "]":
                depth_bracket -= 1
        i += 1

    if last_safe_pos > 0:
        try:
            return json.loads(text[:last_safe_pos])
        except json.JSONDecodeError:
            pass

    # Fallback: strip trailing incomplete value and close
    # Find the last comma at depth 1, cut there, close the object
    depth_brace = 0
    in_string = False
    escape_next = False
    last_comma_at_root = -1

    for i, c in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth_brace += 1
        elif c == "}":
            depth_brace -= 1
        elif c == "," and depth_brace == 1:
            last_comma_at_root = i

    if last_comma_at_root > 0:
        candidate = text[:last_comma_at_root] + "\n}"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def parse_json(text: str) -> dict:
    """
    Robust JSON parser for LLM responses.

    Handles:
    - Pure JSON
    - ```json ... ``` code fences
    - Leading/trailing prose around a JSON object
    - Truncated JSON objects (best-effort repair)
    """
    text = text.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first complete JSON object or array via regex
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    # Truncated JSON repair
    repaired = _try_repair_truncated(text)
    if repaired is not None:
        return repaired

    raise ValueError(f"无法解析 JSON：{text[:200]}")
