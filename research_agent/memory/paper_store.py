"""
PaperStore v3.1  — ChromaDB vector + BM25 + RRF hybrid retrieval

Architecture:
  - ChromaDB  : persistent vector store (OpenAI text-embedding-3-small)
  - rank-bm25 : BM25Okapi lexical retrieval over the same corpus
  - RRF       : Reciprocal Rank Fusion to merge both ranked lists

Chunking strategy (sliding window):
  - Semantic header chunk: Title + Problem + Method + Result (always indexed)
  - Sliding window over full_text: CHUNK_SIZE=600 tokens (~800 chars), OVERLAP=150 chars
    Each window chunk is prefixed with the paper title for retrieval grounding.

Integration points:
  - add_paper(summary, full_text): called after read_papers, indexes paper chunks
  - query(question, top_k):        called by paper_store_query tool, returns merged snippets
  - save() / flush():              persists ChromaDB collection to disk
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Optional

from research_agent.state import Summary

logger = logging.getLogger(__name__)

# ── lazy imports so the module loads even without chromadb installed ──────────

def _get_chroma_client(persist_dir: str):
    import chromadb
    return chromadb.PersistentClient(path=persist_dir)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI text-embedding-3-small."""
    from openai import OpenAI
    client = OpenAI()
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [d.embedding for d in response.data]


# ── helpers ───────────────────────────────────────────────────────────────────

def _doc_id(title: str, chunk_key: str) -> str:
    raw = f"{title}|{chunk_key}"
    return hashlib.md5(raw.encode()).hexdigest()


def _tokenize(text: str) -> list[str]:
    """Simple English tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ── PaperStore ────────────────────────────────────────────────────────────────

class PaperStore:
    """Hybrid paper retrieval store: ChromaDB vector + BM25 + RRF."""

    COLLECTION_NAME = "papers"

    def __init__(self, persist_dir: str = "./paper_index", use_reranker: bool = False):
        self.persist_dir = persist_dir
        self.use_reranker = use_reranker  # toggle cross-encoder reranking
        self._client = None          # chromadb.PersistentClient (lazy)
        self._collection = None      # chromadb.Collection (lazy)
        self._paper_meta: list[dict] = []

        # BM25 state (rebuilt on demand)
        self._bm25_corpus: list[list[str]] = []   # tokenized docs
        self._bm25_ids: list[str] = []            # parallel doc-id list
        self._bm25_texts: list[str] = []          # parallel raw text list
        self._bm25_model = None                   # rank_bm25.BM25Okapi instance
        self._bm25_dirty = True

        self._reranker = None  # lazy-loaded cross-encoder

        self._load_meta()

    # ── init helpers ──────────────────────────────────────────────────────────

    def _ensure_collection(self):
        if self._collection is not None:
            return
        try:
            self._client = _get_chroma_client(self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            # Reload BM25 corpus from persisted collection
            self._reload_bm25_from_chroma()
        except Exception as e:
            logger.warning(f"[PaperStore] ChromaDB init failed: {e}")
            self._collection = None

    def _reload_bm25_from_chroma(self):
        """Sync BM25 corpus from existing ChromaDB documents."""
        if self._collection is None:
            return
        try:
            # Note: "ids" is always returned; only "documents"/"metadatas"/"embeddings" go in include
            all_docs = self._collection.get(include=["documents"])
            ids = all_docs.get("ids", [])
            docs = all_docs.get("documents", [])
            self._bm25_ids = list(ids)
            self._bm25_texts = list(docs)
            self._bm25_corpus = [_tokenize(d) for d in docs]
            self._bm25_model = None
            self._bm25_dirty = False
        except Exception as e:
            logger.warning(f"[PaperStore] BM25 reload failed: {e}")

    # ── metadata persistence ──────────────────────────────────────────────────

    def _meta_path(self) -> str:
        return os.path.join(self.persist_dir, "paper_meta.json")

    def _load_meta(self):
        path = self._meta_path()
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._paper_meta = json.load(f)
            except Exception:
                self._paper_meta = []

    def _save_meta(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(self._paper_meta, f, ensure_ascii=False, indent=2)

    # ── indexing ──────────────────────────────────────────────────────────────

    def add_paper(self, summary: Summary, full_text: str = "") -> bool:
        """Index a paper summary into ChromaDB + BM25. Returns True on success."""
        self._ensure_collection()
        if self._collection is None:
            return False

        title = summary.get("title", "")
        if not title:
            return False

        chunks = self._make_chunks(summary, full_text)
        if not chunks:
            return False

        ids, texts, metadatas = [], [], []
        for chunk_type, text in chunks:
            doc_id = _doc_id(title, chunk_type)
            ids.append(doc_id)
            texts.append(text)
            metadatas.append({"title": title, "chunk_type": chunk_type})

        # Embed and upsert
        try:
            embeddings = _embed_texts(texts)
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.warning(f"[PaperStore] upsert failed for '{title}': {e}")
            return False

        # Update BM25 corpus (incremental)
        existing_set = set(self._bm25_ids)
        for doc_id, text in zip(ids, texts):
            if doc_id not in existing_set:
                self._bm25_ids.append(doc_id)
                self._bm25_texts.append(text)
                self._bm25_corpus.append(_tokenize(text))
        self._bm25_dirty = True

        # Update citation metadata
        if not any(m["title"] == title for m in self._paper_meta):
            self._paper_meta.append({
                "title": title,
                "authors": [],
                "year": 0,
                "worth_reproducing": summary.get("worth_reproducing", False),
            })

        return True

    # Sliding window parameters
    CHUNK_SIZE = 800    # chars per window chunk (~600 tokens)
    CHUNK_OVERLAP = 150  # overlap between consecutive windows

    def _make_chunks(self, summary: Summary, full_text: str) -> list[tuple[str, str]]:
        title = summary.get("title", "")
        problem = summary.get("problem", "") or ""
        method = summary.get("method", "") or ""
        result = summary.get("result", "") or ""
        contributions = summary.get("contributions", []) or []
        limitations = summary.get("limitations", "") or ""

        chunks: list[tuple[str, str]] = []

        # Semantic header chunk — always indexed, highest retrieval signal
        header = (
            f"Title: {title}\n"
            f"Problem: {problem}\n"
            f"Method: {method}\n"
            f"Result: {result}\n"
            + "\n".join(f"- {c}" for c in contributions)
            + (f"\nLimitations: {limitations}" if limitations else "")
        ).strip()
        if header:
            chunks.append(("header", header))

        # Sliding window over full text
        if full_text:
            step = self.CHUNK_SIZE - self.CHUNK_OVERLAP
            for i, start in enumerate(range(0, len(full_text), step)):
                window = full_text[start : start + self.CHUNK_SIZE]
                if not window.strip():
                    continue
                text = f"Title: {title}\n\n{window}"
                chunks.append((f"window_{i}", text))

        return chunks

    # ── reranking ──────────────────────────────────────────────────────────────

    def _get_reranker(self):
        """Lazy-load cross-encoder reranker (BGE-Reranker-Large)."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder("BAAI/bge-reranker-large")
            except Exception as e:
                logger.warning(f"[PaperStore] Failed to load reranker: {e}")
                self._reranker = False  # Mark as unavailable
        return self._reranker if self._reranker is not False else None

    def _rerank_snippets(self, question: str, candidates: dict[str, str]) -> list[str]:
        """
        Rerank candidates using cross-encoder.
        candidates: {doc_id: snippet_text}
        Returns: sorted doc_ids by score (descending)
        """
        if not self.use_reranker:
            return list(candidates.keys())

        reranker = self._get_reranker()
        if reranker is None:
            return list(candidates.keys())

        try:
            pairs = [(question, text[:512]) for text in candidates.values()]
            scores = reranker.predict(pairs)
            ranked = sorted(
                zip(candidates.keys(), scores),
                key=lambda x: x[1],
                reverse=True
            )
            return [doc_id for doc_id, _ in ranked]
        except Exception as e:
            logger.warning(f"[PaperStore] Reranking failed: {e}")
            return list(candidates.keys())

    # ── retrieval ─────────────────────────────────────────────────────────────

    def _ensure_bm25(self):
        if self._bm25_dirty or self._bm25_model is None:
            if self._bm25_corpus:
                from rank_bm25 import BM25Okapi
                self._bm25_model = BM25Okapi(self._bm25_corpus)
            self._bm25_dirty = False

    def _bm25_rank(self, query: str, top_k: int) -> list[str]:
        """Return BM25-ranked doc IDs."""
        self._ensure_bm25()
        if self._bm25_model is None or not self._bm25_ids:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25_model.get_scores(q_tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [self._bm25_ids[i] for i, _ in ranked if scores[i] > 0]

    def query(self, question: str, top_k: int = 6) -> str:
        """
        Hybrid retrieval: ChromaDB vector + BM25 + RRF fusion.
        Returns merged snippets as a string.
        """
        self._ensure_collection()

        vec_ids: list[str] = []
        snippets: dict[str, str] = {}  # id → text snippet

        # 1. Vector retrieval via ChromaDB
        if self._collection is not None and self._collection.count() > 0:
            try:
                q_emb = _embed_texts([question])[0]
                n_results = min(max(top_k * 3, 10), self._collection.count())
                results = self._collection.query(
                    query_embeddings=[q_emb],
                    n_results=n_results,
                    include=["documents", "metadatas"],
                )
                for doc_id, doc, meta in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                ):
                    vec_ids.append(doc_id)
                    title = meta.get("title", "")
                    chunk_type = meta.get("chunk_type", "")
                    snippets[doc_id] = f"[{title}] ({chunk_type})\n{doc[:600]}"
            except Exception as e:
                logger.warning(f"[PaperStore] vector query failed: {e}")

        # 2. BM25 retrieval
        bm25_ids = self._bm25_rank(question, top_k=max(top_k * 3, 10))
        for i, doc_id in enumerate(bm25_ids):
            if doc_id not in snippets and i < len(self._bm25_ids):
                idx = self._bm25_ids.index(doc_id)
                snippets[doc_id] = self._bm25_texts[idx][:600]

        # 3. RRF fusion
        rrf_k = 60.0
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(vec_ids, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, doc_id in enumerate(bm25_ids, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        if not scores:
            return ""

        # Get candidates before reranking for optional cross-encoder pass
        candidate_ids = sorted(scores, key=lambda k: scores[k], reverse=True)[:max(top_k * 2, 12)]
        candidates = {doc_id: snippets[doc_id] for doc_id in candidate_ids if doc_id in snippets}

        # 4. Optional reranking with cross-encoder
        final_ids = self._rerank_snippets(question, candidates)[:top_k]

        return "\n\n---\n\n".join(
            snippets[doc_id] for doc_id in final_ids if doc_id in snippets
        )

    # ── citation export ───────────────────────────────────────────────────────

    def get_citation_list(self) -> list[str]:
        citations = []
        for meta in self._paper_meta:
            title = meta.get("title", "Unknown")
            authors = meta.get("authors", [])
            year = meta.get("year", 0)
            if authors:
                first = authors[0].split()[-1]
                author_str = f"{first} et al." if len(authors) > 1 else first
                ref = f"{author_str}, {year} — {title}" if year else f"{author_str} — {title}"
            else:
                ref = title
            citations.append(ref)
        return citations

    def update_paper_meta(self, title: str, authors: list[str], year: int):
        for meta in self._paper_meta:
            if meta["title"] == title:
                if authors:
                    meta["authors"] = authors
                if year:
                    meta["year"] = year
                return
        self._paper_meta.append({
            "title": title, "authors": authors, "year": year,
            "worth_reproducing": False,
        })

    def get_reproducible_papers(self) -> list[str]:
        return [m["title"] for m in self._paper_meta if m.get("worth_reproducing")]

    # ── persistence ───────────────────────────────────────────────────────────

    def flush(self):
        """No-op: ChromaDB upserts immediately. Kept for API compatibility."""
        pass

    def save(self):
        """Persist metadata (ChromaDB collection persists automatically)."""
        self._save_meta()

    def __len__(self) -> int:
        self._ensure_collection()
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._bm25_ids)
