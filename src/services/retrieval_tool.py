from __future__ import annotations

import re
from typing import Any, List, Tuple
from src.knowledge import knowledge_base
from src.utils.log_utils import setup_logger
import traceback
from src.core.config import config
from src.core.embedding import get_shared_embedder
from src.extraction.graph_builder import load_entity_graph
from src.database.graphrag_retriever import GraphRAGRetriever

logger = setup_logger(__name__)

_get_embedder = get_shared_embedder

# 仅支持三种检索模式：rag | graphrag | both
_ALLOWED_MODES = frozenset({"rag", "graphrag", "both"})


def normalize_retrieval_mode(mode: str | None) -> str:
    """非法或空值回退为 rag。"""
    m = (mode or "rag").strip().lower()
    return m if m in _ALLOWED_MODES else "rag"


def _effective_mode_and_graph_search(mode: str) -> tuple[str, str]:
    """
    mode 已为 normalize 后的 rag / graphrag / both。
    返回 (effective_mode, graph_search_type)，供 GraphRAGRetriever.rerank_chunks 使用。
    """
    if mode == "graphrag":
        return "graphrag", "local"
    if mode == "both":
        return "both", "community"
    return "rag", "local"


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower())


def _build_citation_record(ref: int, chunk: dict, db_id: str) -> dict[str, Any]:
    """供前端悬停展示的 chunk 级引用元数据（与检索顺序 ref 一致）。"""
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


def _format_chunk(content: str, source: str, db_id: str, metadata: dict | None = None) -> str:
    """
    将 chunk 文本连同元数据格式化为 LLM 可直接引用的段落。

    输出格式（示例）：
        ─────────────────────────────────────────
        [Paper ID: arxiv_2305_12345 | Title: Attention Is All You Need | Section: Introduction]
        The Transformer replaces recurrent layers with multi-head self-attention...
        (来源文件: paper.pdf | 知识库: kb_abc123)
        ─────────────────────────────────────────
    """
    meta = metadata or {}

    # 从 metadata 中提取 paper_id / title / section
    paper_id = (
        meta.get("paper_id")
        or meta.get("id")
        or source
    )
    title = (
        meta.get("title")
        or meta.get("paper_title")
        or source
    )
    section = (
        meta.get("section")
        or meta.get("chapter")
        or ""
    )

    # 构建元数据头部
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


def _extract_chunks_from_query_result(query_result: Any, db_id: str) -> list[dict]:
    """
    兼容两类返回结构：
    1) list[{"content","metadata","score"}]
    2) {"documents":...,"metadatas":...}
    """
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
                    "formatted": _format_chunk(content, source, db_id, metadata),
                }
            )
        return chunks

    if isinstance(query_result, dict) and query_result.get("documents"):
        docs = query_result["documents"][0] if isinstance(query_result["documents"][0], list) else query_result["documents"]
        metas = query_result["metadatas"][0] if "metadatas" in query_result and isinstance(query_result["metadatas"][0], list) else query_result.get("metadatas", [{}] * len(docs))
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
                    "formatted": _format_chunk(content, source, db_id, meta),
                }
            )
    return chunks


def _rag_rerank(chunks: list[dict], query_text: str, top_k: int) -> list[dict]:
    query_tokens = set(_tokenize(query_text))
    ranked = []
    for chunk in chunks:
        vector_score = float(chunk.get("score", 0.0) or 0.0)
        content_tokens = set(_tokenize(chunk.get("content", "")))
        overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
        rag_score = 0.8 * vector_score + 0.2 * overlap
        ranked.append({**chunk, "rag_score": rag_score})
    ranked.sort(key=lambda x: x.get("rag_score", 0.0), reverse=True)

    # 结果多样性：同 source 超过2条时适当抑制
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


def _graphrag_rerank(chunks: list[dict], query_text: str, top_k: int) -> list[dict]:
    """
    轻量 GraphRAG：
    - 节点：候选 chunk
    - 边：同 source 强连接 + 关键词重叠连接
    - 得分：向量分 + 图传播分
    """
    if not chunks:
        return []
    query_tokens = set(_tokenize(query_text))
    node_tokens = [set(_tokenize(c.get("content", ""))) for c in chunks]
    n = len(chunks)
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n)]

    for i in range(n):
        source_i = chunks[i].get("source", "")
        for j in range(i + 1, n):
            source_j = chunks[j].get("source", "")
            overlap = len(node_tokens[i] & node_tokens[j])
            weight = 0.0
            if source_i and source_i == source_j:
                weight += 0.35
            if overlap > 0:
                union = max(len(node_tokens[i] | node_tokens[j]), 1)
                weight += overlap / union
            if weight > 0:
                neighbors[i].append((j, weight))
                neighbors[j].append((i, weight))

    init_scores = []
    for i, c in enumerate(chunks):
        v = float(c.get("score", 0.0) or 0.0)
        q_overlap = 0.0
        if query_tokens:
            q_overlap = len(query_tokens & node_tokens[i]) / max(len(query_tokens), 1)
        init_scores.append(0.7 * v + 0.3 * q_overlap)

    scores = init_scores[:]
    for _ in range(2):
        next_scores = scores[:]
        for i in range(n):
            if not neighbors[i]:
                continue
            total_w = sum(w for _, w in neighbors[i]) or 1.0
            propagated = sum(scores[j] * w for j, w in neighbors[i]) / total_w
            next_scores[i] = 0.6 * init_scores[i] + 0.4 * propagated
        scores = next_scores

    ranked = []
    for i, c in enumerate(chunks):
        ranked.append({**c, "graph_score": scores[i]})
    ranked.sort(key=lambda x: x.get("graph_score", 0.0), reverse=True)
    return ranked[:top_k]


def _chunk_paper_key(metadata: dict | None) -> str:
    meta = metadata or {}
    return str(
        meta.get("paper_id")
        or meta.get("id")
        or meta.get("paper_title")
        or ""
    ).strip()


def _graph_guided_rerank(
    chunks: list[dict],
    query_text: str,
    paper_scores: dict[str, float],
    top_k: int,
    fusion_w: float,
) -> list[dict]:
    """
    将向量相似度与图谱论文相关性融合，使图谱参与纯 RAG 的排序决策。
    fused = (1 - w) * vector_score + w * graph_paper_score
    """
    if not paper_scores:
        return _rag_rerank(chunks, query_text, top_k)

    fused_list: list[dict] = []
    for c in chunks:
        base = float(c.get("score", 0.0) or 0.0)
        meta = c.get("metadata") or {}
        pid = _chunk_paper_key(meta)
        g = 0.0
        if pid:
            g = float(paper_scores.get(pid, 0.0))
            if g == 0.0:
                for k, v in paper_scores.items():
                    if k and (k in pid or pid in k):
                        g = max(g, float(v))
        fused_score = (1.0 - fusion_w) * base + fusion_w * g
        fused_list.append(
            {
                **c,
                "graph_paper_score": round(g, 4),
                "fused_score": round(fused_score, 4),
            }
        )
    fused_list.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)

    selected: list[dict] = []
    source_count: dict[str, int] = {}
    for item in fused_list:
        source = item.get("source", "")
        cnt = source_count.get(source, 0)
        if cnt >= 2 and len(selected) < max(top_k - 1, 1):
            continue
        source_count[source] = cnt + 1
        selected.append({**item, "score": item.get("fused_score", item.get("score", 0.0))})
        if len(selected) >= top_k:
            break
    return selected


def _preview_text(text: str, max_len: int = 120) -> str:
    """终端日志用：单行、截断过长问题文本。"""
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _merge_ranked_chunks(primary: list[dict], secondary: list[dict], top_k: int) -> list[dict]:
    merged = []
    seen = set()
    for item in primary + secondary:
        key = f"{item.get('source','')}::{item.get('content','')[:160]}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= top_k:
            break
    return merged


async def retrieval_tool(
    querys: List[str],
    preferred_db_ids: List[str] | None = None,
    top_k: int | None = None,
    retrieval_mode: str = "rag",
) -> Tuple[List[str], List[dict[str, Any]]]:
    """
    检索工具，从向量数据库中查询相关文档 (RAG 切片升级版)

    :param querys: 查询文本列表
    :return: (格式化片段列表, 与顺序一致的 chunk 引用元数据列表)
    """
    retrieval_results: list[str] = []
    citation_records: list[dict[str, Any]] = []

    try:
        k = top_k or config.get("top_k", 5)
        recall_k = int(config.get("retrieval_recall_top_k", 20))
        query_text = " ".join([q for q in querys if q])

        # ==========================================
        # 从用户创建的永久知识库中检索文档（支持多库）
        # ==========================================
        db_ids: List[str] = []

        # 调用方显式传入列表（含空列表）时，仅检索这些库，不回退到全局 config
        if preferred_db_ids is not None:
            db_ids.extend([x for x in preferred_db_ids if x])
        else:
            current_db_ids = config.get("current_db_ids", default=[])
            if isinstance(current_db_ids, list):
                db_ids.extend([x for x in current_db_ids if x])
            current_db_id = config.get("current_db_id", default=None)
            if current_db_id:
                db_ids.append(current_db_id)

        # 去重保持顺序
        seen = set()
        ordered_db_ids = []
        for db_id in db_ids:
            if db_id not in seen:
                seen.add(db_id)
                ordered_db_ids.append(db_id)

        mode = normalize_retrieval_mode(retrieval_mode)
        effective_mode, graph_search_type = _effective_mode_and_graph_search(mode)

        logger.info(
            "[检索] 开始 | mode=%s | graph_search=%s | top_k=%s | 知识库=%s | 问题=%s",
            effective_mode,
            graph_search_type,
            k,
            ordered_db_ids if ordered_db_ids else "(未指定，将跳过向量检索)",
            _preview_text(query_text),
        )

        if not ordered_db_ids:
            logger.warning("[检索] 结束：未指定任何知识库 db_id，无检索结果")
            return [], []

        ref_global = 1
        for db_id in ordered_db_ids:
            query_top_k = recall_k
            logger.info(
                "[检索] → 知识库 %s | Chroma 首次召回条数上限=%s",
                db_id,
                query_top_k,
            )
            db_results = await knowledge_base.aquery(
                querys,
                db_id=db_id,
                top_k=query_top_k,
                similarity_threshold=config.get("similarity_threshold", 0.0)
            )
            chunks = _extract_chunks_from_query_result(db_results, db_id)
            logger.info(
                "[检索]   %s | 向量库原始命中 %s 条 chunk",
                db_id,
                len(chunks),
            )
            rerank_note = ""
            if effective_mode == "graphrag":
                graph = load_entity_graph(db_id)
                if graph:
                    chunks = GraphRAGRetriever(graph, _get_embedder()).rerank_chunks(
                        chunks, query_text, k, search_type=graph_search_type
                    )
                    rerank_note = f"GraphRAGRetriever rerank (search_type={graph_search_type})"
                else:
                    chunks = _graphrag_rerank(chunks, query_text, k)
                    rerank_note = "无实体图谱，回退轻量 GraphRAG 重排"
            elif effective_mode == "both":
                rag_chunks = _rag_rerank(chunks, query_text, k)
                graph = load_entity_graph(db_id)
                if graph:
                    graph_chunks = GraphRAGRetriever(graph, _get_embedder()).rerank_chunks(
                        chunks, query_text, k, search_type=graph_search_type
                    )
                    rerank_note = f"RAG + GraphRAG 合并 (search_type={graph_search_type})"
                else:
                    graph_chunks = _graphrag_rerank(chunks, query_text, k)
                    rerank_note = "RAG + 轻量图重排（无实体图谱文件）"
                chunks = _merge_ranked_chunks(rag_chunks, graph_chunks, k)
            else:
                # 纯 RAG：若存在实体图谱，用语义种子 → 论文相关性参与排序决策
                graph = load_entity_graph(db_id, embedder=_get_embedder())
                if graph:
                    retriever = GraphRAGRetriever(graph, _get_embedder())
                    paper_scores = retriever.get_paper_relevance_scores(query_text)
                    fusion_w = float(config.get("graphrag.rag_graph_fusion_weight") or 0.35)
                    if paper_scores:
                        chunks = _graph_guided_rerank(
                            chunks, query_text, paper_scores, k, fusion_w
                        )
                        rerank_note = f"向量+图谱论文相关性融合 (fusion_w={fusion_w})"
                    else:
                        chunks = _rag_rerank(chunks, query_text, k)
                        rerank_note = "纯 RAG 重排（图谱无论文相关分）"
                else:
                    chunks = _rag_rerank(chunks, query_text, k)
                    rerank_note = "纯 RAG 重排（无实体图谱）"

            logger.info(
                "[检索]   %s | 重排后 %s 条 | %s",
                db_id,
                len(chunks),
                rerank_note,
            )
            for rank, ch in enumerate(chunks[: min(8, len(chunks))], start=1):
                src = str(ch.get("source", ""))[:80]
                sc = float(ch.get("score", 0.0) or 0.0)
                logger.info(
                    "[检索]     #%s score=%.4f 来源=%s",
                    rank,
                    sc,
                    src or "(无)",
                )

            for chunk in chunks:
                citation_records.append(_build_citation_record(ref_global, chunk, db_id))
                retrieval_results.append(chunk["formatted"])
                ref_global += 1

        logger.info(
            "[检索] 完成 | 共输出 %s 条片段（可能含多个知识库累加）",
            len(retrieval_results),
        )
        return retrieval_results, citation_records

    except Exception as e:
        logger.error(f"知识库检索失败 {e}, {traceback.format_exc()}")
        return [], []