from __future__ import annotations

import json
import sys
from typing import Any

from research_agent.tools import get_all_specs, get_tool


def _ok(req_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_request(payload: dict) -> dict:
    req_id = payload.get("id")
    method = payload.get("method", "")
    params = payload.get("params", {}) or {}

    if method in {"initialize"}:
        return _ok(
            req_id,
            {
                "serverInfo": {"name": "research-agent-tools", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        )
    if method in {"tools/list", "list_tools"}:
        return _ok(req_id, {"tools": get_all_specs("mcp")})
    if method in {"tools/call", "call_tool"}:
        name = params.get("name", "")
        arguments = params.get("arguments", {}) or {}
        if not name:
            return _err(req_id, -32602, "missing tool name")
        try:
            tool = get_tool(name)
        except KeyError as e:
            return _err(req_id, -32601, str(e))
        try:
            text = tool.run(**arguments)
            return _ok(req_id, {"content": [{"type": "text", "text": text}]})
        except Exception as e:
            return _err(req_id, -32000, f"tool execution failed: {e}")
    if method in {"shutdown", "exit"}:
        return _ok(req_id, {"ok": True})

    return _err(req_id, -32601, f"unknown method: {method}")


def serve_stdio():
    """
    简化版 MCP stdio loop（JSON-RPC over newline-delimited JSON）。
    """
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            sys.stdout.write(json.dumps(_err(None, -32700, "parse error"), ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_request(payload)
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()

        if payload.get("method") in {"shutdown", "exit"}:
            break
