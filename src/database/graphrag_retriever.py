"""
graphrag_retriever.py — GraphRAG 五分量检索重排器

职责：
  - rerank_chunks()  : 五分量融合评分（vec / graph / entity / embedding / lexical）
  - global_search()  : 全图传播（3 跳），适合跨领域总结型查询
  - local_search()   : 1 跳局部检索，适合精确实体定位
  - get_community_context() : 社区摘要上下文提取
  - get_multi_hop_paths()   : 多跳推理路径提取

时间复杂度优化（Optimization 3）：
  原实现在 `for chunk in chunks:` 内部每次调用图传播，导致重复计算。
  优化后：
    1. [O(V+E)]  先全局一次性计算所有节点的最终分数（图传播 + 社区 Boost + 实体命中 + 嵌入相似度）
    2. [O(N)]    再遍历 chunks，通过 paper_id → entity_ids 索引直接映射节点分数
  总复杂度：O(V+E) + O(N)，而非原来的 O(N*(V+E))。

依赖注入：
  GraphRAGRetriever(graph_data, embedder)
  — graph_data : 图谱字典
  — embedder   : VectorEmbedder 实例（用于查询向量化）
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.core.config import config
from src.core.embedding import VectorEmbedder, embedding_cosine_similarity
from src.extraction.graph_schema import (
    RELATION_LABELS as _RELATION_LABELS,
    tokenize as _tokenize,
    normalize_text as _normalize_text,
    entity_id as _entity_id,
    expand_abbreviation as _expand_abbreviation,
    type_boost as _type_boost,
)
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


# ============================================================
# GraphRAGRetriever
# ============================================================

class GraphRAGRetriever:
    """
    GraphRAG 五分量检索重排器。

    五分量评分公式：
        final_score =
            0.35 * vector_score       （向量相似度，来自向量库）
            0.25 * graph_score        （图传播分数）
            0.15 * entity_match_score （实体精确命中 F1）
            0.15 * embedding_score    （图节点语义向量相似度）
            0.10 * lexical_score      （词法 overlap）

    Args:
        graph_data: 图谱字典（nodes / edges / paper_entities / communities）。
        embedder:   VectorEmbedder 实例，用于查询文本向量化。
    """

    def __init__(
        self,
        graph_data: dict[str, Any],
        embedder: VectorEmbedder,
    ) -> None:
        self.graph   = graph_data
        self.embedder = embedder

    # ------------------------------------------------------------------
    # 公共接口：供上层 Agent 统一调用
    # ------------------------------------------------------------------

    def local_search(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        局部检索：1 跳图传播，精确实体定位。

        Args:
            query:  查询文本。
            chunks: 向量库返回的候选 chunk 列表。
            top_k:  返回数量。

        Returns:
            重排后的 top_k chunks（已附加各分量分数）。
        """
        return self.rerank_chunks(
            chunks=chunks,
            query_text=query,
            top_k=top_k,
            search_type="local",
        )

    def global_search(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        全局检索：3 跳图传播 + 社区 Boost，适合跨领域总结型查询。

        Args:
            query:  查询文本。
            chunks: 向量库返回的候选 chunk 列表。
            top_k:  返回数量。

        Returns:
            重排后的 top_k chunks（已附加各分量分数）。
        """
        return self.rerank_chunks(
            chunks=chunks,
            query_text=query,
            top_k=top_k,
            search_type="global",
        )

    # ------------------------------------------------------------------
    # 核心重排方法（时间复杂度优化版本）
    # ------------------------------------------------------------------

    def rerank_chunks(
        self,
        chunks: list[dict[str, Any]],
        query_text: str,
        top_k: int,
        search_type: str = "local",
    ) -> list[dict[str, Any]]:
        """
        GraphRAG 五分量检索重排（时间复杂度优化版本）。

        优化点：
          原始实现在 for chunk 循环内重复计算图节点分数，复杂度 O(N*(V+E))。
          本实现将所有图节点分数计算提前到循环外，分两阶段：
            Phase 1 [O(V+E)]：计算全局节点分数（seed + 图传播 + 社区 Boost）
                              以及节点级 entity_match_score / embedding_score。
            Phase 2 [O(N)]：  遍历 chunks，通过 paper_id 索引直接映射节点分数。
          总复杂度：O(V+E) + O(N)。

        Args:
            chunks:      向量库返回的候选 chunk 列表（含 score / metadata / content）。
            query_text:  查询文本。
            top_k:       返回数量。
            search_type: "local" | "community" | "global"。

        Returns:
            重排并截取 top_k 的 chunk 列表。
        """
        if not chunks or not self.graph:
            return chunks[:top_k]

        graph          = self.graph
        nodes          = graph.get("nodes", {})
        edges          = graph.get("edges", [])
        paper_entities = graph.get("paper_entities", {})
        communities    = graph.get("communities", {})

        query_tokens = set(_tokenize(query_text))
        query_embed  = self.embedder.get_embedding(query_text)

        # --------------------------------------------------------
        # Phase 1a：构建邻接表
        # --------------------------------------------------------
        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in edges:
            s = edge.get("source")
            t = edge.get("target")
            w = float(edge.get("weight", 1.0))
            if s and t:
                adjacency[s].append((t, w))
                adjacency[t].append((s, w))

        # --------------------------------------------------------
        # Phase 1b：种子分数（三阶段查询实体提取）
        # --------------------------------------------------------
        seed_scores = self._extract_query_entity_seeds(query_text)

        # --------------------------------------------------------
        # Phase 1c：实体命中分数（entity_match_score）— 全局一次计算
        # --------------------------------------------------------
        entity_match_scores: dict[str, float] = {}
        for node_id, node in nodes.items():
            norm_label   = node.get("norm_label", "")
            label_tokens = set(_tokenize(norm_label))
            if query_tokens and label_tokens:
                precision = len(query_tokens & label_tokens) / len(label_tokens)
                recall    = len(query_tokens & label_tokens) / len(query_tokens)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    entity_match_scores[node_id] = f1 * _type_boost(node.get("type", ""))

        # --------------------------------------------------------
        # Phase 1d：图节点嵌入相似度（embedding_score）— 全局一次计算
        # --------------------------------------------------------
        embedding_scores: dict[str, float] = {}
        for node_id, node in nodes.items():
            node_embed = node.get("embedding")
            if node_embed:
                embedding_scores[node_id] = embedding_cosine_similarity(query_embed, node_embed)

        # --------------------------------------------------------
        # Phase 1e：社区 Boost — 全局一次计算
        # --------------------------------------------------------
        community_boost: dict[str, float] = {}
        if search_type in ("community", "global") and communities:
            for _comm_id, comm in communities.items():
                comm_summary = comm.get("summary", "")
                comm_name    = comm.get("community_name", "")
                comm_tokens  = set(_tokenize(comm_summary + " " + comm_name))
                if not comm_tokens:
                    comm_tokens = set(_tokenize(" ".join(comm.get("key_entities", []))))
                if query_tokens:
                    overlap = len(query_tokens & comm_tokens) / max(len(query_tokens), 1)
                    if overlap > 0:
                        for nid in comm.get("nodes", []):
                            community_boost[nid] = max(
                                community_boost.get(nid, 0.0),
                                overlap * 0.8,
                            )

        # --------------------------------------------------------
        # Phase 1f：图传播（Personalized PageRank-like propagation）
        # 先将 community_boost 叠加到 seed_scores，然后迭代传播
        # --------------------------------------------------------
        hops  = 3 if search_type == "global" else 1
        alpha = float(config.get("graphrag.seed_alpha") or 0.55)

        node_scores: dict[str, float] = dict(seed_scores)
        for nid, boost in community_boost.items():
            node_scores[nid] = node_scores.get(nid, 0.0) + boost

        for _ in range(max(hops, 1)):
            next_scores = dict(node_scores)
            for node_id, neighs in adjacency.items():
                if not neighs:
                    continue
                total_w    = sum(w for _, w in neighs) or 1.0
                propagated = sum(node_scores.get(n, 0.0) * w for n, w in neighs) / total_w
                seed       = seed_scores.get(node_id, 0.0)
                next_scores[node_id] = alpha * seed + (1 - alpha) * propagated
            node_scores = next_scores

        # --------------------------------------------------------
        # Phase 2：O(N) 遍历 chunks，直接映射预计算的节点分数
        # --------------------------------------------------------
        ranked: list[dict[str, Any]] = []
        for chunk in chunks:
            vec_score      = float(chunk.get("score", 0.0) or 0.0)
            metadata       = chunk.get("metadata", {}) or {}
            content        = chunk.get("content", "")
            content_tokens = set(_tokenize(content))
            lexical_score  = (
                len(query_tokens & content_tokens) / max(len(query_tokens), 1)
                if query_tokens else 0.0
            )

            graph_score  = 0.0
            entity_score = 0.0
            emb_score    = 0.0
            score_count  = 0

            # 通过 paper_id 索引映射实体分数
            paper_id_raw = str(metadata.get("paper_id", "")).strip()
            if paper_id_raw and paper_id_raw in paper_entities:
                ent_ids = paper_entities[paper_id_raw]
                g_vals  = [node_scores.get(eid, 0.0) for eid in ent_ids]
                e_vals  = [entity_match_scores.get(eid, 0.0) for eid in ent_ids]
                b_vals  = [embedding_scores.get(eid, 0.0) for eid in ent_ids]
                if g_vals:
                    graph_score  = sum(g_vals) / len(g_vals)
                    entity_score = max(e_vals) if e_vals else 0.0
                    emb_score    = sum(b_vals) / len(b_vals) if b_vals else 0.0
                    score_count += 1

            # 方法名直接命中
            method_name = str(metadata.get("methodology_name", "")).strip()
            if method_name:
                mid = _entity_id("Method", method_name)
                if mid in node_scores and node_scores[mid] > 0:
                    graph_score  = max(graph_score,  node_scores[mid])
                    entity_score = max(entity_score, entity_match_scores.get(mid, 0.0))
                    emb_score    = max(emb_score,    embedding_scores.get(mid, 0.0))
                    score_count += 1

            # 任务/问题直接命中
            core_problem = str(metadata.get("core_problem", "")).strip()
            if core_problem:
                tid = _entity_id("Task", core_problem[:100])
                if tid in node_scores and node_scores[tid] > 0:
                    graph_score  = max(graph_score,  node_scores[tid])
                    entity_score = max(entity_score, entity_match_scores.get(tid, 0.0))
                    emb_score    = max(emb_score,    embedding_scores.get(tid, 0.0))
                    score_count += 1

            if score_count > 0:
                graph_score = min(graph_score, 1.0)

            # 五分量融合
            final_score = (
                0.35 * vec_score
                + 0.25 * graph_score
                + 0.15 * entity_score
                + 0.15 * emb_score
                + 0.10 * lexical_score
            )

            ranked.append({
                **chunk,
                "graph_score":   round(graph_score, 4),
                "entity_score":  round(entity_score, 4),
                "emb_score":     round(emb_score, 4),
                "lexical_score": round(lexical_score, 4),
                "final_score":   round(final_score, 4),
                "search_type":   search_type,
            })

        ranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # 社区上下文
    # ------------------------------------------------------------------

    def get_community_context(
        self,
        query_text: str,
        top_n: int = 3,
    ) -> str:
        """
        提取与查询最相关的社区摘要，格式化为 LLM 可读文本块。
        若社区无 LLM 摘要，则使用语义标签 + 关键实体列表作为 fallback。
        """
        graph       = self.graph
        communities = graph.get("communities", {})
        if not communities:
            return ""

        query_tokens = set(_tokenize(query_text))
        query_embed  = self.embedder.get_embedding(query_text)
        scored: list[tuple[float, str, str, str]] = []

        for comm_id, comm in communities.items():
            summary   = comm.get("summary", "")
            comm_name = comm.get("community_name") or comm.get("semantic_tag") or f"Community {comm_id}"
            sem_tag   = comm.get("semantic_tag", "")

            if not summary:
                summary = (
                    f"[{sem_tag}] "
                    "Key entities: " + ", ".join(comm.get("key_entities", [])[:10])
                )

            comm_tokens  = set(_tokenize(summary + " " + comm_name))
            text_overlap = (
                len(query_tokens & comm_tokens) / max(len(query_tokens), 1)
                if query_tokens else 0.0
            )

            embed_sim = 0.0
            node_ids  = comm.get("nodes", [])
            nodes     = graph.get("nodes", {})
            sample_embeds = [
                nodes[nid]["embedding"]
                for nid in node_ids[:8]
                if nid in nodes and "embedding" in nodes[nid]
            ]
            if sample_embeds:
                avg_sim   = sum(embedding_cosine_similarity(query_embed, e) for e in sample_embeds)
                embed_sim = avg_sim / len(sample_embeds)

            combined_score = 0.6 * text_overlap + 0.4 * embed_sim
            scored.append((combined_score, comm_name, summary, sem_tag))

        scored.sort(reverse=True)
        parts = []
        for _, name, summary, _ in scored[:top_n]:
            if summary:
                parts.append(f"[Research Community: {name}]\n{summary}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # 多跳推理路径
    # ------------------------------------------------------------------

    def get_multi_hop_paths(
        self,
        query_text: str,
        max_hops: int = 3,
    ) -> str:
        """
        多跳推理路径提取：从查询实体出发，找到最具信息量的路径链。

        Returns:
            格式化为 [Multi-hop Reasoning Paths] 的字符串。
        """
        seed_scores = self._extract_query_entity_seeds(query_text)
        if not seed_scores:
            return ""

        graph  = self.graph
        nodes  = graph.get("nodes", {})
        edges  = graph.get("edges", [])

        out_edges: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for e in edges:
            src, tgt, rel = e.get("source"), e.get("target"), e.get("type", "related_to")
            if src and tgt:
                out_edges[src].append((rel, tgt, str(e.get("weight", 0.5))))

        top_seeds   = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        all_paths: list[tuple[float, list[str]]] = []

        for start_node, seed_score in top_seeds:
            stack: list[tuple[str, list[str], float]] = [(start_node, [start_node], seed_score)]
            while stack:
                current, path, path_score = stack.pop()
                if len(path) > 1:
                    all_paths.append((path_score, path))
                if len(path) >= max_hops + 1:
                    continue
                for rel, neighbor, w_str in out_edges.get(current, []):
                    if neighbor in path:
                        continue
                    edge_w    = float(w_str) if w_str else 0.5
                    new_score = path_score * edge_w * _type_boost(
                        nodes.get(neighbor, {}).get("type", "")
                    )
                    if new_score < 0.05:
                        continue
                    path_repr = path + [f"--[{rel}]-->", neighbor]
                    stack.append((neighbor, path_repr, new_score))

        all_paths.sort(key=lambda x: x[0], reverse=True)
        seen_paths: set[str] = set()
        lines = ["[Multi-hop Reasoning Paths]"]
        path_count = 0
        for _, path in all_paths:
            if path_count >= 5:
                break
            readable_parts = []
            for segment in path:
                if segment.startswith("--["):
                    rel_name = segment[3:-3]
                    readable_parts.append(f" {_RELATION_LABELS.get(rel_name, rel_name)} ")
                else:
                    label = nodes.get(segment, {}).get("label", segment)
                    readable_parts.append(label)
            readable = "→".join(readable_parts)
            if readable not in seen_paths:
                seen_paths.add(readable)
                lines.append(f"  Path {path_count + 1}: {readable}")
                path_count += 1

        return "\n".join(lines) if path_count > 0 else ""

    # ------------------------------------------------------------------
    # 局部子图上下文（供 LLM 直接引用）
    # ------------------------------------------------------------------

    def get_local_subgraph_context(
        self,
        query_text: str,
        max_hops: int = 2,
        max_triples: int = 40,
    ) -> str:
        """
        提取与查询最相关实体的局部子图，转换为结构化可读知识块。

        输出示例::

            [Local Knowledge Subgraph]

              [Method: Transformer]
                Description: A model architecture using self-attention...
                → proposes              : Attention Mechanism (Concept)
                → evaluated on          : WMT14 (Dataset)
                → achieves              : BLEU 28.4 (Result)

        Args:
            query_text:   查询文本。
            max_hops:     从种子节点出发 BFS 扩展的最大跳数。
            max_triples:  子图中保留的最大边数。

        Returns:
            格式化的子图文本，若无相关实体则返回空字符串。
        """
        seed_scores = self._extract_query_entity_seeds(query_text)
        if not seed_scores:
            return ""

        graph = self.graph
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", [])

        top_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        seed_ids  = {nid for nid, _ in top_seeds}

        visited  = set(seed_ids)
        frontier = set(seed_ids)
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for edge in edges:
                s, t = edge["source"], edge["target"]
                if s in frontier and t not in visited:
                    next_frontier.add(t)
                elif t in frontier and s not in visited:
                    next_frontier.add(s)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        subgraph_edges = [
            e for e in edges
            if e["source"] in visited and e["target"] in visited
        ][:max_triples]

        if not subgraph_edges:
            return ""

        out_edges: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for e in subgraph_edges:
            src_id    = e["source"]
            dst_label = nodes.get(e["target"], {}).get("label", e["target"])
            rel_label = _RELATION_LABELS.get(e["type"], e["type"])
            dst_type  = nodes.get(e["target"], {}).get("type", "")
            out_edges[src_id].append((rel_label, dst_label, dst_type))

        lines = ["[Local Knowledge Subgraph]"]
        rendered_nodes: set[str] = set()
        ordered_nodes = list(seed_ids) + [n for n in visited if n not in seed_ids]

        for node_id in ordered_nodes:
            if node_id not in out_edges or node_id in rendered_nodes:
                continue
            rendered_nodes.add(node_id)

            node_info  = nodes.get(node_id, {})
            node_label = node_info.get("label", node_id)
            node_type  = node_info.get("type", "Unknown")
            desc       = node_info.get("description", "")

            lines.append(f"\n  [{node_type}: {node_label}]")
            if desc:
                lines.append(f"    Description: {desc[:120]}")
            for rel_label, dst_label, dst_type in out_edges[node_id][:8]:
                type_hint = f" ({dst_type})" if dst_type else ""
                lines.append(f"    → {rel_label:<20}: {dst_label}{type_hint}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 私有辅助：查询实体种子提取
    # ------------------------------------------------------------------

    def _extract_query_entity_seeds(self, query_text: str) -> dict[str, float]:
        """
        三阶段种子提取：
          Stage 1: 别名精确匹配
          Stage 2: 缩写展开后匹配
          Stage 3: Token overlap 补充
        """
        graph        = self.graph
        query_norm   = _normalize_text(query_text)
        query_tokens = set(_tokenize(query_norm))
        aliases      = graph.get("entity_aliases", {})
        nodes        = graph.get("nodes", {})
        seeds: dict[str, float] = {}

        # Stage 1: 别名短语精确命中
        for alias, node_id in aliases.items():
            if alias and alias in query_norm:
                node_type = nodes.get(node_id, {}).get("type", "")
                seeds[node_id] = max(seeds.get(node_id, 0.0), 1.0 * _type_boost(node_type))

        # Stage 2: 缩写展开后命中
        for token in list(query_tokens):
            expanded = _expand_abbreviation(token)
            if expanded != token:
                exp_node_id = aliases.get(expanded)
                if exp_node_id:
                    node_type = nodes.get(exp_node_id, {}).get("type", "")
                    seeds[exp_node_id] = max(
                        seeds.get(exp_node_id, 0.0),
                        0.9 * _type_boost(node_type),
                    )

        # Stage 3: Token overlap 补充
        for node_id, node in nodes.items():
            label_tokens = set(_tokenize(node.get("norm_label", "")))
            if not label_tokens:
                continue
            overlap = (
                len(query_tokens & label_tokens) / max(len(query_tokens), 1)
                if query_tokens else 0.0
            )
            if overlap > 0:
                seeds[node_id] = max(
                    seeds.get(node_id, 0.0),
                    overlap * _type_boost(node.get("type", "")),
                )

        return seeds

    def get_paper_relevance_scores(self, query_text: str) -> dict[str, float]:
        """
        根据查询在图谱中的实体种子，为每篇论文（paper_entities 键）计算相关性分数 [0,1]。
        用于纯 RAG 路径下将图谱参与检索排序决策（与向量分融合）。
        """
        seed_scores = self._extract_query_entity_seeds(query_text)
        if not seed_scores:
            return {}

        paper_entities = self.graph.get("paper_entities", {})
        paper_scores: dict[str, float] = {}

        for paper_id_raw, ent_ids in paper_entities.items():
            best = 0.0
            for eid in ent_ids:
                best = max(best, float(seed_scores.get(eid, 0.0)))
            if best > 0:
                paper_scores[str(paper_id_raw)] = min(1.0, best)

        return paper_scores
