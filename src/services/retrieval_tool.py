from __future__ import annotations

import re
from typing import Any, List
from src.knowledge.knowledge import knowledge_base
from src.utils.log_utils import setup_logger
import traceback
from src.core.config import config
from src.services.graph_store import load_entity_graph, rerank_chunks_by_entity_graph

logger = setup_logger(__name__)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower())


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


def _hybrid_rerank(chunks: list[dict], query_text: str, top_k: int) -> list[dict]:
    query_tokens = set(_tokenize(query_text))
    if not query_tokens:
        return sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]

    ranked = []
    for chunk in chunks:
        vector_score = float(chunk.get("score", 0.0) or 0.0)
        content_tokens = set(_tokenize(chunk.get("content", "")))
        overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
        hybrid_score = 0.7 * vector_score + 0.3 * overlap
        ranked.append({**chunk, "hybrid_score": hybrid_score})
    ranked.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return ranked[:top_k]


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
) -> List[str]:
    """
    检索工具，从向量数据库中查询相关文档 (RAG 切片升级版)

    :param querys: 查询文本列表
    :return: 包含文档内容的字符串列表
    """
    retrieval_results = []

    try:
        k = top_k or config.get("top_k", 5)
        query_text = " ".join([q for q in querys if q])

        # ==========================================
        # 1. 从临时知识库 (RAG 论文切片) 中检索文档
        # ==========================================
        tmp_db_id = config.get("tmp_db_id")
        if tmp_db_id:
            tmpdb_results = await knowledge_base.aquery(
                querys,
                db_id=tmp_db_id,
                top_k=config.get("tmpdb_top_k", k),
                similarity_threshold=config.get("tmpdb_similarity_threshold", 0.0)
            )
            tmp_chunks = _extract_chunks_from_query_result(tmpdb_results, tmp_db_id)
            tmp_k = config.get("tmpdb_top_k", k)
            _tmp_graphrag_type_map = {
                "graphrag": "local",
                "graphrag_local": "local",
                "graphrag_community": "community",
                "graphrag_global": "global",
            }
            if retrieval_mode in _tmp_graphrag_type_map:
                tmp_search_type = _tmp_graphrag_type_map[retrieval_mode]
                graph = load_entity_graph(tmp_db_id)
                if graph:
                    tmp_chunks = rerank_chunks_by_entity_graph(tmp_chunks, graph, query_text, tmp_k, search_type=tmp_search_type)
                else:
                    tmp_chunks = _graphrag_rerank(tmp_chunks, query_text, tmp_k)
            elif retrieval_mode == "both":
                rag_chunks = _rag_rerank(tmp_chunks, query_text, tmp_k)
                graph = load_entity_graph(tmp_db_id)
                if graph:
                    graph_chunks = rerank_chunks_by_entity_graph(tmp_chunks, graph, query_text, tmp_k, search_type="community")
                else:
                    graph_chunks = _graphrag_rerank(tmp_chunks, query_text, tmp_k)
                tmp_chunks = _merge_ranked_chunks(rag_chunks, graph_chunks, tmp_k)
            else:
                tmp_chunks = _rag_rerank(tmp_chunks, query_text, tmp_k)

            for chunk in tmp_chunks:
                retrieval_results.append(chunk["formatted"])

        # ==========================================
        # 2. 从用户创建的长期知识库中检索文档（支持多库）
        # ==========================================
        db_ids: List[str] = []

        # 优先使用调用方显式指定
        if preferred_db_ids:
            db_ids.extend([x for x in preferred_db_ids if x])

        # 其次兼容全局单选与多选配置
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

        for db_id in ordered_db_ids:
            db_info = knowledge_base.get_database_info(db_id) or {}
            additional = db_info.get("additional_params", {}) or {}
            retrieval_method = additional.get("retrieval_method", "vector")
            # 将前端扩展模式映射到内部 effective_mode 和 search_type
            _graphrag_search_type_map = {
                "graphrag_local": ("graphrag", "local"),
                "graphrag_community": ("graphrag", "community"),
                "graphrag_global": ("graphrag", "global"),
                "graphrag": ("graphrag", "local"),
                "both": ("both", "community"),
                "rag": ("rag", "local"),
            }
            if retrieval_mode in _graphrag_search_type_map:
                effective_mode, graph_search_type = _graphrag_search_type_map[retrieval_mode]
            elif retrieval_method == "hybrid":
                effective_mode, graph_search_type = "both", "community"
            else:
                effective_mode, graph_search_type = "rag", "local"

            query_top_k = k * 4 if effective_mode in {"graphrag", "both"} else k
            db_results = await knowledge_base.aquery(
                querys,
                db_id=db_id,
                top_k=query_top_k,
                similarity_threshold=config.get("similarity_threshold", 0.0)
            )
            chunks = _extract_chunks_from_query_result(db_results, db_id)
            if effective_mode == "graphrag":
                graph = load_entity_graph(db_id)
                if graph:
                    chunks = rerank_chunks_by_entity_graph(chunks, graph, query_text, k, search_type=graph_search_type)
                else:
                    chunks = _graphrag_rerank(chunks, query_text, k)
            elif effective_mode == "both":
                rag_chunks = _rag_rerank(chunks, query_text, k)
                graph = load_entity_graph(db_id)
                if graph:
                    graph_chunks = rerank_chunks_by_entity_graph(chunks, graph, query_text, k, search_type=graph_search_type)
                else:
                    graph_chunks = _graphrag_rerank(chunks, query_text, k)
                chunks = _merge_ranked_chunks(rag_chunks, graph_chunks, k)
            elif retrieval_method == "hybrid":
                # 保留旧 hybrid（向量 + 词重排）兼容
                chunks = _hybrid_rerank(chunks, query_text, k)
            else:
                chunks = _rag_rerank(chunks, query_text, k)

            for chunk in chunks:
                retrieval_results.append(chunk["formatted"])

        return retrieval_results

    except Exception as e:
        logger.error(f"知识库检索失败 {e}, {traceback.format_exc()}")
        return [f"检索失败: {e}"]