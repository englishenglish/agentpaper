"""
纯向量 RAG：Chroma 召回、chunk 解析与轻量重排（全部封装在 ``RagRetriever`` 中）。
"""
from __future__ import annotations

import re
from typing import Any, List

from src.core.embedding import VectorEmbedder, get_shared_embedder
from src.rag import knowledge_base
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_get_embedder = get_shared_embedder


class RagRetriever:
    """
    纯向量 RAG：从 Chroma 召回 chunk，解析为结构化条目，并可做轻量 RAG 重排。

    分词、格式化、解析、重排、引用元数据等均实现为本类方法；实体图谱由
    ``GraphRAGRetriever`` 处理，``HybridRetriever`` 负责编排。
    """

    def __init__(self, embedder: VectorEmbedder | None = None) -> None:
        self._embedder = embedder

    @property
    def embedder(self) -> VectorEmbedder:
        return self._embedder if self._embedder is not None else _get_embedder()

    # ------------------------------------------------------------------
    # 文本工具
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower())

    @staticmethod
    def format_chunk(content: str, source: str, db_id: str, metadata: dict | None = None) -> str:
        meta = metadata or {}
        paper_id = meta.get("paper_id") or meta.get("id") or source
        title = meta.get("title") or meta.get("paper_title") or source
        section = meta.get("section") or meta.get("chapter") or ""
        header_parts = [f"Paper ID: {paper_id}", f"Title: {title}"]
        if section:
            header_parts.append(f"Section: {section}")
        header = " | ".join(header_parts)
        return (
            f"─────────────────────────────────────────\n"
            f"[{header}]\n"
            f"{content}\n"
            f"(来源文件: {source} | 知识库: {db_id})\n"
            f"─────────────────────────────────────────"
        )

    @classmethod
    def extract_chunks_from_query_result(cls, query_result: Any, db_id: str) -> list[dict]:
        chunks: list[dict] = []
        if isinstance(query_result, list):
            for item in query_result:
                content = str(item.get("content", ""))
                metadata = item.get("metadata", {}) or {}
                score = float(item.get("score", 0.0) or 0.0)
                source = metadata.get("source", "未知来源")
                chunks.append(
                    {
                        "content": content,
                        "source": source,
                        "score": score,
                        "metadata": metadata,
                        "formatted": cls.format_chunk(content, source, db_id, metadata),
                    }
                )
            return chunks

        if isinstance(query_result, dict) and query_result.get("documents"):
            docs = (
                query_result["documents"][0]
                if isinstance(query_result["documents"][0], list)
                else query_result["documents"]
            )
            metas = (
                query_result["metadatas"][0]
                if "metadatas" in query_result and isinstance(query_result["metadatas"][0], list)
                else query_result.get("metadatas", [{}] * len(docs))
            )
            dists = query_result.get("distances", [])
            if dists and isinstance(dists[0], list):
                dists = dists[0]
            for idx, (doc, meta) in enumerate(zip(docs, metas)):
                source = meta.get("source", "未知来源")
                distance = dists[idx] if idx < len(dists) else 0.0
                score = 1 - float(distance or 0.0)
                content = str(doc)
                chunks.append(
                    {
                        "content": content,
                        "source": source,
                        "score": score,
                        "metadata": meta,
                        "formatted": cls.format_chunk(content, source, db_id, meta),
                    }
                )
        return chunks

    @classmethod
    def rag_rerank(cls, chunks: list[dict], query_text: str, top_k: int) -> list[dict]:
        query_tokens = set(cls.tokenize(query_text))
        ranked = []
        for chunk in chunks:
            vector_score = float(chunk.get("score", 0.0) or 0.0)
            content_tokens = set(cls.tokenize(chunk.get("content", "")))
            overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
            rag_score = 0.8 * vector_score + 0.2 * overlap
            ranked.append({**chunk, "rag_score": rag_score})
        ranked.sort(key=lambda x: x.get("rag_score", 0.0), reverse=True)

        selected = []
        source_count: dict[str, int] = {}
        for item in ranked:
            source = item.get("source", "")
            cnt = source_count.get(source, 0)
            if cnt >= 2 and len(selected) < max(top_k - 1, 1):
                continue
            source_count[source] = cnt + 1
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def build_citation_record(ref: int, chunk: dict, db_id: str) -> dict[str, Any]:
        meta = chunk.get("metadata") or {}
        content = chunk.get("content", "") or ""
        preview = content[:520] + ("…" if len(content) > 520 else "")
        cid = meta.get("chunk_id") or meta.get("id")
        if not cid:
            cid = f"{db_id}_ref{ref}"
        return {
            "ref": ref,
            "chunk_id": str(cid),
            "paper_id": str(meta.get("paper_id") or meta.get("id") or ""),
            "title": str(meta.get("title") or meta.get("paper_title") or ""),
            "section": str(meta.get("section") or meta.get("chapter") or ""),
            "source": str(chunk.get("source", "") or ""),
            "db_id": db_id,
            "score": round(float(chunk.get("score", 0.0) or 0.0), 4),
            "preview": preview,
        }

    # ------------------------------------------------------------------
    # 异步召回与实例级重排
    # ------------------------------------------------------------------

    async def fetch_chunks(
        self,
        querys: List[str],
        db_id: str,
        recall_k: int,
        similarity_threshold: float,
    ) -> list[dict]:
        db_results = await knowledge_base.aquery(
            querys,
            db_id=db_id,
            top_k=recall_k,
            similarity_threshold=similarity_threshold,
        )
        return self.extract_chunks_from_query_result(db_results, db_id)

    def rerank(self, chunks: list[dict], query_text: str, top_k: int) -> list[dict]:
        return self.rag_rerank(chunks, query_text, top_k)
