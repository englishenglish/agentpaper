"""
混合检索：编排 ``RagRetriever``（向量）与 ``GraphRAGRetriever``（实体图谱）。

支持三种检索模式（``retrieval_mode``）：

- **rag**：向量召回后以 RAG 重排为主；若存在实体图谱可融合论文级图谱相关性。
- **graphrag**：以 ``GraphRAGRetriever`` 五分量重排为主；无图时回退轻量图式重排。
- **both**：并行得到 RAG 排序与 GraphRAG 排序，再合并去重至 ``top_k``。
"""
from __future__ import annotations

import traceback
from typing import Any, List, Tuple

from src.core.config import config
from src.core.embedding import VectorEmbedder
from src.graphrag.graph_builder import load_entity_graph
from src.utils.log_utils import setup_logger

from src.retriever.graphrag_retriever import GraphRAGRetriever
from src.retriever.rag_retriever import RagRetriever

logger = setup_logger(__name__)

MODE_RAG = "rag"
MODE_GRAPHRAG = "graphrag"
MODE_BOTH = "both"
_ALLOWED_MODES = frozenset({MODE_RAG, MODE_GRAPHRAG, MODE_BOTH})


def normalize_retrieval_mode(mode: str | None) -> str:
    """非法或空值回退为 ``rag``。"""
    m = (mode or MODE_RAG).strip().lower()
    return m if m in _ALLOWED_MODES else MODE_RAG


def _effective_mode_and_graph_search(mode: str) -> tuple[str, str]:
    if mode == MODE_GRAPHRAG:
        return MODE_GRAPHRAG, "local"
    if mode == MODE_BOTH:
        return MODE_BOTH, "community"
    return MODE_RAG, "local"


def _graphrag_rerank(chunks: list[dict], query_text: str, top_k: int) -> list[dict]:
    if not chunks:
        return []
    query_tokens = set(RagRetriever.tokenize(query_text))
    node_tokens = [set(RagRetriever.tokenize(c.get("content", ""))) for c in chunks]
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
    if not paper_scores:
        return RagRetriever.rag_rerank(chunks, query_text, top_k)

    fused_list: list[dict] = []
    for c in chunks:
        base = float(c.get("score", 0.0) or 0.0)
        meta = c.get("metadata") or {}
        pid = _chunk_paper_key(meta)
        g = 0.0
        if pid:
            g = float(paper_scores.get(pid, 0.0))
            if g == 0.0:
                for pk, v in paper_scores.items():
                    if pk and (pk in pid or pid in pk):
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


class HybridRetriever:
    """
    混合检索：统一入口 ``retrieve``，支持三种模式（见模块文档）。

    便捷方法：``retrieve_rag`` / ``retrieve_graphrag`` / ``retrieve_both`` 等价于
    传入对应 ``retrieval_mode`` 的 ``retrieve``。
    """

    MODE_RAG = MODE_RAG
    MODE_GRAPHRAG = MODE_GRAPHRAG
    MODE_BOTH = MODE_BOTH

    def __init__(self, rag: RagRetriever | None = None) -> None:
        self.rag = rag or RagRetriever()

    async def retrieve_rag(
        self,
        querys: List[str],
        preferred_db_ids: List[str] | None = None,
        top_k: int | None = None,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        """仅 RAG 路径（向量 + 轻量重排，可选图谱论文融合）。"""
        return await self.retrieve(querys, preferred_db_ids, top_k, retrieval_mode=MODE_RAG)

    async def retrieve_graphrag(
        self,
        querys: List[str],
        preferred_db_ids: List[str] | None = None,
        top_k: int | None = None,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        """以 GraphRAG 五分量重排为主。"""
        return await self.retrieve(querys, preferred_db_ids, top_k, retrieval_mode=MODE_GRAPHRAG)

    async def retrieve_both(
        self,
        querys: List[str],
        preferred_db_ids: List[str] | None = None,
        top_k: int | None = None,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        """RAG 与 GraphRAG 各排一套，合并去重。"""
        return await self.retrieve(querys, preferred_db_ids, top_k, retrieval_mode=MODE_BOTH)

    def _rerank_graphrag_branch(
        self,
        chunks: list[dict],
        query_text: str,
        k: int,
        graph_search_type: str,
        db_id: str,
        emb: VectorEmbedder,
    ) -> tuple[list[dict], str]:
        graph = load_entity_graph(db_id)
        if graph:
            out = GraphRAGRetriever(graph, emb).rerank_chunks(
                chunks, query_text, k, search_type=graph_search_type
            )
            return out, f"GraphRAGRetriever rerank (search_type={graph_search_type})"
        out = _graphrag_rerank(chunks, query_text, k)
        return out, "无实体图谱，回退轻量 GraphRAG 重排"

    def _rerank_both_branch(
        self,
        chunks: list[dict],
        query_text: str,
        k: int,
        graph_search_type: str,
        db_id: str,
        emb: VectorEmbedder,
    ) -> tuple[list[dict], str]:
        rag_chunks = self.rag.rerank(chunks, query_text, k)
        graph = load_entity_graph(db_id)
        if graph:
            graph_chunks = GraphRAGRetriever(graph, emb).rerank_chunks(
                chunks, query_text, k, search_type=graph_search_type
            )
            note = f"RAG + GraphRAG 合并 (search_type={graph_search_type})"
        else:
            graph_chunks = _graphrag_rerank(chunks, query_text, k)
            note = "RAG + 轻量图重排（无实体图谱文件）"
        return _merge_ranked_chunks(rag_chunks, graph_chunks, k), note

    def _rerank_rag_branch(
        self,
        chunks: list[dict],
        query_text: str,
        k: int,
        db_id: str,
        emb: VectorEmbedder,
    ) -> tuple[list[dict], str]:
        graph = load_entity_graph(db_id, embedder=emb)
        if graph:
            g_ret = GraphRAGRetriever(graph, emb)
            paper_scores = g_ret.get_paper_relevance_scores(query_text)
            fusion_w = float(config.get("graphrag.rag_graph_fusion_weight") or 0.35)
            if paper_scores:
                out = _graph_guided_rerank(chunks, query_text, paper_scores, k, fusion_w)
                return out, f"向量+图谱论文相关性融合 (fusion_w={fusion_w})"
            out = self.rag.rerank(chunks, query_text, k)
            return out, "纯 RAG 重排（图谱无论文相关分）"
        out = self.rag.rerank(chunks, query_text, k)
        return out, "纯 RAG 重排（无实体图谱）"

    async def retrieve(
        self,
        querys: List[str],
        preferred_db_ids: List[str] | None = None,
        top_k: int | None = None,
        retrieval_mode: str = MODE_RAG,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        """
        统一检索入口。

        :param retrieval_mode: ``rag`` | ``graphrag`` | ``both``（也可用 ``HybridRetriever.MODE_*``）。
        """
        retrieval_results: list[str] = []
        citation_records: list[dict[str, Any]] = []

        try:
            k = top_k or config.get("top_k", 5)
            recall_k = int(config.get("retrieval_recall_top_k", 20))
            query_text = " ".join([q for q in querys if q])

            db_ids: List[str] = []
            if preferred_db_ids is not None:
                db_ids.extend([x for x in preferred_db_ids if x])
            else:
                current_db_ids = config.get("current_db_ids", default=[])
                if isinstance(current_db_ids, list):
                    db_ids.extend([x for x in current_db_ids if x])
                current_db_id = config.get("current_db_id", default=None)
                if current_db_id:
                    db_ids.append(current_db_id)

            seen_ids = set()
            ordered_db_ids: list[str] = []
            for db_id in db_ids:
                if db_id not in seen_ids:
                    seen_ids.add(db_id)
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

            sim_th = float(config.get("similarity_threshold", 0.0))
            ref_global = 1
            emb = self.rag.embedder

            for db_id in ordered_db_ids:
                query_top_k = recall_k
                logger.info(
                    "[检索] → 知识库 %s | Chroma 首次召回条数上限=%s",
                    db_id,
                    query_top_k,
                )
                chunks = await self.rag.fetch_chunks(querys, db_id, query_top_k, sim_th)
                logger.info(
                    "[检索]   %s | 向量库原始命中 %s 条 chunk",
                    db_id,
                    len(chunks),
                )

                if effective_mode == MODE_GRAPHRAG:
                    chunks, rerank_note = self._rerank_graphrag_branch(
                        chunks, query_text, k, graph_search_type, db_id, emb
                    )
                elif effective_mode == MODE_BOTH:
                    chunks, rerank_note = self._rerank_both_branch(
                        chunks, query_text, k, graph_search_type, db_id, emb
                    )
                else:
                    chunks, rerank_note = self._rerank_rag_branch(
                        chunks, query_text, k, db_id, emb
                    )

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
                    citation_records.append(RagRetriever.build_citation_record(ref_global, chunk, db_id))
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


async def retrieval_tool(
    querys: List[str],
    preferred_db_ids: List[str] | None = None,
    top_k: int | None = None,
    retrieval_mode: str = MODE_RAG,
) -> Tuple[List[str], List[dict[str, Any]]]:
    """向后兼容：委托 ``HybridRetriever.retrieve``。"""
    return await HybridRetriever().retrieve(
        querys,
        preferred_db_ids=preferred_db_ids,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )
