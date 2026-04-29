from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import xml.etree.ElementTree as ET

from github import Github, GithubException
import requests

from research_agent.llm.router import call_llm, system
from research_agent.memory.conversation_memory import ConversationMemory
from research_agent.memory.paper_store import PaperStore
from research_agent.tools.base import BaseTool
from research_agent.tools.registry import register_tool


def _search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    try:
        encoded = requests.utils.quote(query)
        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query=all:{encoded}&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        )
        resp = requests.get(url, timeout=20, headers={"User-Agent": "research-agent/1.0"})
        resp.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        papers: list[dict] = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", "", ns) or "").strip()
            link = ""
            for lnk in entry.findall("atom:link", ns):
                if lnk.attrib.get("type") == "text/html":
                    link = lnk.attrib.get("href", "")
                    break
            if not link:
                id_elem = entry.find("atom:id", ns)
                link = id_elem.text if id_elem is not None else ""
            published = entry.findtext("atom:published", "2020", ns) or "2020"
            year = int(published[:4])
            authors = [
                (a.findtext("atom:name", "", ns) or "").strip()
                for a in entry.findall("atom:author", ns)
            ][:5]
            if title:
                papers.append({
                    "title": title,
                    "abstract": summary[:500],
                    "url": link,
                    "pdf_path": "",
                    "year": year,
                    "authors": authors,
                    "citations": 0,
                })
        return papers[:max_results]
    except Exception:
        return []


def _search_semantic_scholar(query: str, max_results: int = 10) -> list[dict]:
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": min(max_results, 100),
                "fields": "title,year,externalIds,citationCount,abstract,authors",
            },
            timeout=15,
            headers={"User-Agent": "research-agent/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        papers: list[dict] = []
        for item in (data.get("data", []) or [])[:max_results]:
            arxiv_id = str(((item.get("externalIds") or {}).get("ArXiv") or "")).strip()
            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
            title = str(item.get("title", "") or "").strip()
            if not title:
                continue
            papers.append({
                "title": title,
                "abstract": str(item.get("abstract", "") or "")[:500],
                "url": url,
                "pdf_path": "",
                "year": int(item.get("year", 0) or 0),
                "authors": [a.get("name", "") for a in (item.get("authors") or [])][:5],
                "citations": int(item.get("citationCount", 0) or 0),
            })
        return papers
    except Exception:
        return []


def _project_root() -> str:
    return os.path.abspath(os.getcwd())


def _safe_text_read(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_search_results(html: str, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for href, title in re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', html, flags=re.I | re.S):
        clean_title = _strip_html(title)
        if not clean_title or href in seen:
            continue
        seen.add(href)
        results.append({"title": clean_title[:180], "url": href})
        if len(results) >= limit:
            break
    return results


@register_tool
class DoneTool(BaseTool):
    name = "done"
    description = "结束当前任务，并返回最终结果。"
    input_schema = {
        "properties": {
            "result": {"type": "string", "description": "给用户的最终答复"},
            "confidence": {"type": "string", "description": "high|medium|low", "default": "medium"},
        },
        "required": ["result"],
    }

    def run(self, **kwargs: Any) -> str:
        payload = {
            "result": str(kwargs.get("result", "")).strip(),
            "confidence": str(kwargs.get("confidence", "medium")).strip().lower() or "medium",
        }
        return json.dumps(payload, ensure_ascii=False)


@register_tool
class SessionContextTool(BaseTool):
    name = "session_context"
    description = "读取会话短长记忆与压缩上下文。"
    input_schema = {
        "properties": {
            "session_id": {"type": "string", "description": "会话 ID"},
            "char_budget": {"type": "integer", "default": 5000},
        },
        "required": ["session_id"],
    }

    def run(self, **kwargs: Any) -> str:
        session_id = str(kwargs.get("session_id", "default"))
        char_budget = int(kwargs.get("char_budget", 5000))
        mem = ConversationMemory(session_id)
        payload = mem.build_context_package(current_input="", char_budget=char_budget)
        return json.dumps(payload, ensure_ascii=False)


@register_tool
class ListLocalFilesTool(BaseTool):
    name = "list_local_files"
    description = "列出本地目录下的文本文件，适合先摸清项目结构。"
    input_schema = {
        "properties": {
            "path": {"type": "string", "description": "目录路径，默认当前工作目录", "default": "."},
            "limit": {"type": "integer", "default": 40},
        },
        "required": [],
    }

    def run(self, **kwargs: Any) -> str:
        base = str(kwargs.get("path", ".") or ".").strip()
        limit = max(1, min(int(kwargs.get("limit", 40)), 200))
        root = os.path.abspath(os.path.expanduser(base))
        if not os.path.exists(root):
            return json.dumps({"files": [], "count": 0, "error": "path not found"}, ensure_ascii=False)

        files: list[str] = []
        if os.path.isfile(root):
            files = [root]
        else:
            for current_root, _, names in os.walk(root):
                for name in names:
                    if name.startswith("."):
                        continue
                    files.append(os.path.join(current_root, name))
                    if len(files) >= limit:
                        break
                if len(files) >= limit:
                    break
        return json.dumps(
            {
                "files": [os.path.relpath(path, _project_root()) for path in files[:limit]],
                "count": len(files[:limit]),
                "root": root,
            },
            ensure_ascii=False,
        )


@register_tool
class ReadLocalFileTool(BaseTool):
    name = "read_local_file"
    description = "读取本地文本文件内容。"
    input_schema = {
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "max_chars": {"type": "integer", "default": 4000},
        },
        "required": ["path"],
    }

    def run(self, **kwargs: Any) -> str:
        path = str(kwargs.get("path", "")).strip()
        max_chars = max(200, min(int(kwargs.get("max_chars", 4000)), 20000))
        if not path:
            return json.dumps({"path": "", "content": "", "error": "missing path"}, ensure_ascii=False)
        resolved = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(resolved):
            return json.dumps({"path": resolved, "content": "", "error": "file not found"}, ensure_ascii=False)
        content = _safe_text_read(resolved, max_chars)
        return json.dumps(
            {
                "path": resolved,
                "content": content,
                "truncated": len(content) >= max_chars,
            },
            ensure_ascii=False,
        )


@register_tool
class SaveTextArtifactTool(BaseTool):
    name = "save_text_artifact"
    description = "将文本保存到 output 目录，便于后续继续编辑或交付。"
    input_schema = {
        "properties": {
            "filename": {"type": "string", "description": "输出文件名，如 summary.md"},
            "content": {"type": "string", "description": "文件内容"},
        },
        "required": ["filename", "content"],
    }

    def run(self, **kwargs: Any) -> str:
        filename = os.path.basename(str(kwargs.get("filename", "")).strip())
        content = str(kwargs.get("content", ""))
        if not filename:
            return json.dumps({"path": "", "error": "missing filename"}, ensure_ascii=False)
        output_dir = os.path.abspath("./output")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"path": path, "bytes": len(content.encode("utf-8"))}, ensure_ascii=False)


@register_tool
class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = "抓取指定 URL 的文本内容，适合用户已经给出具体网页时阅读。"
    input_schema = {
        "properties": {
            "url": {"type": "string", "description": "要抓取的 URL"},
            "max_chars": {"type": "integer", "default": 5000},
        },
        "required": ["url"],
    }

    def run(self, **kwargs: Any) -> str:
        url = str(kwargs.get("url", "")).strip()
        max_chars = max(300, min(int(kwargs.get("max_chars", 5000)), 20000))
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return json.dumps({"url": url, "content": "", "error": "unsupported url scheme"}, ensure_ascii=False)
        try:
            response = requests.get(url, timeout=15, headers={"User-Agent": "research-agent/1.0"})
            response.raise_for_status()
        except Exception as e:
            return json.dumps({"url": url, "content": "", "error": str(e)}, ensure_ascii=False)

        text = response.text or ""
        content_type = response.headers.get("content-type", "")
        if "html" in content_type.lower():
            text = _strip_html(text)
        return json.dumps(
            {
                "url": url,
                "content": text[:max_chars],
                "truncated": len(text) > max_chars,
            },
            ensure_ascii=False,
        )


@register_tool
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "执行通用网页搜索，返回标题和 URL 结果。"
    input_schema = {
        "properties": {
            "query": {"type": "string", "description": "搜索 query"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def run(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        limit = max(1, min(int(kwargs.get("limit", 5)), 10))
        if not query:
            return json.dumps({"results": [], "count": 0, "error": "missing query"}, ensure_ascii=False)
        try:
            resp = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                timeout=15,
                headers={"User-Agent": "research-agent/1.0"},
            )
            resp.raise_for_status()
            results = _extract_search_results(resp.text, limit)
            return json.dumps({"results": results, "count": len(results), "query": query}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"results": [], "count": 0, "query": query, "error": str(e)}, ensure_ascii=False)


@register_tool
class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = "仅在 arXiv 搜索论文，返回标题、年份、URL。"
    input_schema = {
        "properties": {
            "query": {"type": "string", "description": "检索 query"},
            "limit": {"type": "integer", "default": 8},
        },
        "required": ["query"],
    }

    def run(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        limit = max(1, min(int(kwargs.get("limit", 8)), 20))
        if not query:
            return json.dumps({"papers": [], "count": 0, "error": "missing query"}, ensure_ascii=False)
        papers = _search_arxiv(query, max_results=limit)
        out = [
            {
                "title": p.get("title", ""),
                "year": p.get("year", 0),
                "url": p.get("url", ""),
                "citations": p.get("citations", 0),
            }
            for p in papers[:limit]
        ]
        return json.dumps({"papers": out, "count": len(out), "query": query}, ensure_ascii=False)


@register_tool
class PaperSearchTool(BaseTool):
    name = "paper_search"
    description = "在 arXiv 与 Semantic Scholar 搜索论文。"
    input_schema = {
        "properties": {
            "query": {"type": "string", "description": "检索 query"},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }

    def run(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        limit = max(1, min(int(kwargs.get("limit", 10)), 50))
        if not query:
            return json.dumps({"papers": [], "count": 0}, ensure_ascii=False)

        arxiv = _search_arxiv(query, max_results=min(limit, 25))
        ss = _search_semantic_scholar(query, max_results=min(limit, 25))
        papers = arxiv + ss
        papers.sort(key=lambda p: (p.get("citations", 0), p.get("year", 0)), reverse=True)
        out = []
        seen = set()
        for p in papers:
            t = (p.get("title", "") or "").strip().lower()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(
                {
                    "title": p.get("title", ""),
                    "year": p.get("year", 0),
                    "citations": p.get("citations", 0),
                    "url": p.get("url", ""),
                }
            )
            if len(out) >= limit:
                break
        return json.dumps({"papers": out, "count": len(out), "query": query}, ensure_ascii=False)


@register_tool
class DownloadPaperTool(BaseTool):
    """Download an arXiv paper PDF, extract full text, and index into local RAG store."""

    name = "download_paper"
    description = (
        "Download a paper PDF from arXiv (given its URL or arXiv ID), extract full text, "
        "and index it into the local paper store for RAG retrieval. "
        "Use this after paper_search to get deeper content beyond abstracts."
    )
    input_schema = {
        "properties": {
            "arxiv_id_or_url": {
                "type": "string",
                "description": "arXiv ID (e.g. '2309.06180') or full URL (e.g. 'https://arxiv.org/abs/2309.06180')",
            },
            "title": {
                "type": "string",
                "description": "Paper title (used as the document key in the store)",
            },
            "paper_store_dir": {
                "type": "string",
                "default": "./paper_index",
                "description": "Local RAG index directory (default: ./paper_index)",
            },
        },
        "required": ["arxiv_id_or_url", "title"],
    }

    _PDF_DIR = "./output/papers"

    @staticmethod
    def _parse_arxiv_id(raw: str) -> str:
        """Extract bare arXiv ID from URL or raw string."""
        raw = raw.strip()
        # e.g. https://arxiv.org/abs/2309.06180  or  https://arxiv.org/pdf/2309.06180v2
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v\d+)?)", raw, re.IGNORECASE)
        if m:
            return m.group(1)
        # bare ID like "2309.06180" or "2309.06180v2"
        m = re.match(r"^(\d{4}\.\d{4,5}(?:v\d+)?)$", raw)
        if m:
            return m.group(1)
        return raw  # return as-is, will fail gracefully

    def run(self, **kwargs: Any) -> str:
        raw = str(kwargs.get("arxiv_id_or_url", "")).strip()
        title = str(kwargs.get("title", "")).strip() or raw
        store_dir = str(kwargs.get("paper_store_dir", self._PDF_DIR)).strip() or "./paper_index"

        arxiv_id = self._parse_arxiv_id(raw)
        if not arxiv_id:
            return json.dumps({"ok": False, "error": "Could not parse arXiv ID"})

        # Strip version suffix for clean filename
        bare_id = re.sub(r"v\d+$", "", arxiv_id)
        os.makedirs(self._PDF_DIR, exist_ok=True)
        pdf_path = os.path.join(self._PDF_DIR, f"{bare_id.replace('/', '_')}.pdf")

        # Download PDF
        if not os.path.exists(pdf_path):
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            try:
                resp = requests.get(pdf_url, timeout=30, headers={"User-Agent": "research-agent/1.0"})
                resp.raise_for_status()
                if "pdf" not in resp.headers.get("content-type", "").lower() and len(resp.content) < 10_000:
                    return json.dumps({"ok": False, "error": f"Unexpected content-type from {pdf_url}"})
                with open(pdf_path, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                return json.dumps({"ok": False, "error": f"Download failed: {e}"})

        # Extract full text
        try:
            from research_agent.tools.pdf_tools import extract_text
            full_text = extract_text(pdf_path, max_chars=15_000)
        except Exception as e:
            full_text = ""

        if not full_text:
            return json.dumps({"ok": False, "error": "PDF downloaded but text extraction failed", "path": pdf_path})

        # Index into PaperStore
        summary = {
            "title": title,
            "problem": "",
            "method": "",
            "result": "",
            "contributions": [],
            "limitations": "",
            "worth_reproducing": False,
        }
        try:
            store = PaperStore(persist_dir="./paper_index")
            store.add_paper(summary, full_text=full_text)
            store.save()
            indexed = True
        except Exception as e:
            indexed = False

        return json.dumps({
            "ok": True,
            "arxiv_id": arxiv_id,
            "title": title,
            "pdf_path": pdf_path,
            "text_chars": len(full_text),
            "indexed": indexed,
        }, ensure_ascii=False)


@register_tool
class PaperStoreQueryTool(BaseTool):
    name = "paper_store_query"
    description = (
        "Query the local paper knowledge base using hybrid retrieval (ChromaDB vector + BM25 + RRF + optional reranker). "
        "Papers are indexed automatically after read_papers. Use this to retrieve relevant evidence by question."
    )
    input_schema = {
        "properties": {
            "question": {"type": "string", "description": "Retrieval query / research question"},
            "top_k": {"type": "integer", "default": 6, "description": "Number of results to return (max 15)"},
            "paper_store_dir": {"type": "string", "default": "./paper_index", "description": "Path to local index (default: ./paper_index)"},
            "use_reranker": {"type": "boolean", "default": False, "description": "Enable cross-encoder reranking (slower but higher precision)"},
        },
        "required": ["question"],
    }

    def run(self, **kwargs: Any) -> str:
        paper_store_dir = str(kwargs.get("paper_store_dir", "./paper_index")).strip() or "./paper_index"
        question = str(kwargs.get("question", "")).strip()
        top_k = max(1, min(int(kwargs.get("top_k", 6)), 15))
        use_reranker = bool(kwargs.get("use_reranker", False))
        if not question:
            return ""
        store = PaperStore(persist_dir=paper_store_dir, use_reranker=use_reranker)
        if len(store) == 0:
            return json.dumps({"result": "", "note": "paper_store is empty — run read_papers first"})
        result = store.query(question, top_k=top_k)
        return json.dumps({"result": result, "count": len(store)}, ensure_ascii=False)


@register_tool
class GithubRepoSearchTool(BaseTool):
    name = "github_repo_search"
    description = "搜索 GitHub 仓库（按 stars 排序）。"
    input_schema = {
        "properties": {
            "query": {"type": "string", "description": "仓库检索词"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def run(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        limit = max(1, min(int(kwargs.get("limit", 5)), 20))
        if not query:
            return json.dumps({"repos": [], "count": 0}, ensure_ascii=False)
        g = Github()
        repos = []
        try:
            for repo in g.search_repositories(query=query, sort="stars", order="desc")[:limit]:
                repos.append(
                    {
                        "full_name": repo.full_name,
                        "url": repo.html_url,
                        "stars": repo.stargazers_count,
                        "description": repo.description or "",
                    }
                )
        except GithubException as e:
            return json.dumps({"repos": [], "count": 0, "error": str(e)}, ensure_ascii=False)
        return json.dumps({"repos": repos, "count": len(repos), "query": query}, ensure_ascii=False)


@register_tool
class ExplainWithEvidenceTool(BaseTool):
    name = "explain_with_evidence"
    description = "基于给定证据回答概念问题（不额外检索）。"
    input_schema = {
        "properties": {
            "question": {"type": "string", "description": "用户问题"},
            "evidence": {"type": "string", "description": "已知证据文本"},
        },
        "required": ["question", "evidence"],
    }

    def run(self, **kwargs: Any) -> str:
        question = str(kwargs.get("question", "")).strip()
        evidence = str(kwargs.get("evidence", "")).strip()
        if not question:
            return "请提供问题。"
        if not evidence:
            return "当前缺少可用证据，建议先检索或读取资料。"

        prompt = f"""你是研究助手。仅基于给定证据回答问题，不编造事实。
问题：{question}
证据：
{evidence[:4500]}

请用中文给出简明回答，结构：
1) 直接答案
2) 关键机制/原因
3) 证据边界（哪里不确定）
"""
        try:
            return call_llm("summarization", [system(prompt)], max_tokens=7000).strip()
        except Exception:
            return "基于现有证据，问题与多智能体中的协同机制相关，但当前无法稳定生成完整解释。"
