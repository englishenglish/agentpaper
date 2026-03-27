"""
community_builder.py — 社区聚类与 LLM 摘要生成器

职责：
  - Louvain 社区聚类（build_communities）
  - 启发式语义标签推断（_infer_community_semantic_tag）
  - 异步批量 LLM 社区摘要生成（generate_summaries）

依赖注入：
  CommunityBuilder(graph_data, llm_client=None)
  — graph_data : 已构建的图谱字典（必须包含 nodes / edges / communities 键）
  — llm_client : 支持 async chat.completions.create 的 LLM 客户端（可选）
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from src.core.config import config
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


class CommunityBuilder:
    """
    社区聚类与摘要生成器。

    设计原则：
      - 聚类逻辑（Louvain）完全在内存中运行，无外部依赖。
      - LLM 摘要通过注入的 llm_client 异步批量生成，不硬编码模型创建逻辑。
      - build_communities / generate_summaries 均返回 self.graph（in-place），
        便于链式调用：builder.build_communities().graph。

    Args:
        graph_data: 含 nodes/edges 的图谱字典（通常是 GraphBuilder.graph）。
        llm_client: 可选，支持 `await llm_client.chat.completions.create(...)` 的客户端。
    """

    def __init__(
        self,
        graph_data: dict[str, Any],
        llm_client: Any = None,
    ) -> None:
        self.graph      = graph_data
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # 公共方法：Louvain 聚类
    # ------------------------------------------------------------------

    def build_communities(self) -> dict[str, Any]:
        """
        对 self.graph 执行 Louvain 社区检测，结果写入 graph["communities"]。

        算法要点：
          1. 构建加权邻接表
          2. 迭代 Louvain 模块度增益（最多 10 轮）
          3. 合并孤立单节点社区到邻居社区
          4. 为每个社区推断语义标签

        Returns:
            self.graph（in-place 修改，含更新后的 communities 字段）
        """
        graph = self.graph
        nodes = list(graph["nodes"].keys())
        if not nodes:
            graph["communities"] = {}
            return graph

        adj: dict[str, dict[str, float]] = defaultdict(dict)
        for edge in graph["edges"]:
            s, t, w = edge["source"], edge["target"], float(edge.get("weight", 1.0))
            adj[s][t] = adj[s].get(t, 0) + w
            adj[t][s] = adj[t].get(s, 0) + w

        community: dict[str, int] = {n: i for i, n in enumerate(nodes)}
        total_weight = sum(w for nbrs in adj.values() for w in nbrs.values()) / 2 or 1.0

        sigma_tot: dict[int, float] = {}
        for n in nodes:
            c = community[n]
            sigma_tot[c] = sigma_tot.get(c, 0.0) + sum(adj[n].values())

        def _modularity_gain(node: str, from_comm: int, to_comm: int) -> float:
            k_i = sum(adj[node].values())
            k_i_in_to   = sum(w for nbr, w in adj[node].items() if community[nbr] == to_comm)
            k_i_in_from = sum(w for nbr, w in adj[node].items() if community[nbr] == from_comm and nbr != node)
            sigma_to   = sigma_tot.get(to_comm, 0.0)
            sigma_from = sigma_tot.get(from_comm, 0.0) - k_i
            gain_join  = k_i_in_to / total_weight - (sigma_to * k_i) / (2 * total_weight ** 2)
            gain_leave = k_i_in_from / total_weight - (sigma_from * k_i) / (2 * total_weight ** 2)
            return gain_join - gain_leave

        for _ in range(10):
            improved = False
            for node in nodes:
                current_comm = community[node]
                best_comm, best_gain = current_comm, 0.0
                neighbor_comms = {community[nbr] for nbr in adj.get(node, {})} - {current_comm}
                for nc in neighbor_comms:
                    gain = _modularity_gain(node, current_comm, nc)
                    if gain > best_gain:
                        best_gain, best_comm = gain, nc
                if best_comm != current_comm:
                    k_i = sum(adj[node].values())
                    sigma_tot[current_comm] = sigma_tot.get(current_comm, 0.0) - k_i
                    sigma_tot[best_comm]    = sigma_tot.get(best_comm, 0.0) + k_i
                    community[node] = best_comm
                    improved = True
            if not improved:
                break

        unique_comms = sorted(set(community.values()))
        remap = {old: new for new, old in enumerate(unique_comms)}
        community = {n: remap[c] for n, c in community.items()}

        comm_nodes: dict[int, list[str]] = defaultdict(list)
        for node, comm_id in community.items():
            comm_nodes[comm_id].append(node)

        # 合并孤立单节点社区
        singleton_ids = [cid for cid, nids in comm_nodes.items() if len(nids) == 1]
        if singleton_ids and len(comm_nodes) > len(singleton_ids):
            largest_comm = max(
                (cid for cid, nids in comm_nodes.items() if len(nids) > 1),
                key=lambda cid: len(comm_nodes[cid]),
                default=None,
            )
            if largest_comm is not None:
                for scid in singleton_ids:
                    node_id = comm_nodes[scid][0]
                    nbr_comms = [
                        community[nbr]
                        for nbr in adj.get(node_id, {})
                        if community[nbr] != scid
                    ]
                    target = nbr_comms[0] if nbr_comms else largest_comm
                    comm_nodes[target].append(node_id)
                    del comm_nodes[scid]

        communities: dict[str, dict] = {}
        node_data = graph["nodes"]
        for new_id, (_, node_ids) in enumerate(comm_nodes.items()):
            labels     = [node_data[nid]["label"] for nid in node_ids if nid in node_data]
            types      = [node_data[nid]["type"]  for nid in node_ids if nid in node_data]
            key_papers = [
                node_data[nid]["label"]
                for nid in node_ids
                if nid in node_data and node_data[nid]["type"] == "Paper"
            ]

            type_counter: dict[str, int] = {}
            for t in types:
                type_counter[t] = type_counter.get(t, 0) + 1
            dominant_types = sorted(type_counter, key=lambda x: -type_counter[x])[:3]

            semantic_tag = self._infer_community_semantic_tag(dominant_types, labels)

            communities[str(new_id)] = {
                "nodes":          node_ids,
                "key_entities":   labels[:12],
                "key_papers":     key_papers[:5],
                "dominant_types": dominant_types,
                "semantic_tag":   semantic_tag,
                "summary":        "",
                "community_name": semantic_tag,
            }

        graph["communities"] = communities
        logger.info(f"Louvain 聚类完成：{len(communities)} 个社区，{len(nodes)} 个节点")
        return graph

    # ------------------------------------------------------------------
    # 公共方法：异步批量 LLM 摘要生成
    # ------------------------------------------------------------------

    async def generate_summaries(self, batch_size: int = 4) -> dict[str, Any]:
        """
        异步批量为所有无摘要社区调用 llm_client 生成语义摘要。

        Args:
            batch_size: 每批并发社区数量。

        Returns:
            self.graph（in-place 修改，含 community summary）

        Raises:
            RuntimeError: 若 llm_client 未在 __init__ 中注入。
        """
        if self.llm_client is None:
            raise RuntimeError(
                "generate_summaries 需要 llm_client，请在 CommunityBuilder.__init__ 中注入。"
            )

        from src.core.prompts import community_summary_prompt  # type: ignore

        graph      = self.graph
        communities = graph.get("communities", {})
        pending = [
            (comm_id, comm)
            for comm_id, comm in communities.items()
            if not comm.get("summary")
        ]

        if not pending:
            return graph

        async def _summarize_one(comm_id: str, comm: dict) -> None:
            key_entities   = comm.get("key_entities", [])
            key_papers     = comm.get("key_papers", [])
            dominant_types = comm.get("dominant_types", [])
            entity_list    = "\n".join(f"- {e}" for e in key_entities[:15])
            prompt = (
                community_summary_prompt
                + f"\n\n【社区信息】\n"
                f"主导实体类型：{', '.join(dominant_types)}\n"
                f"关键实体：\n{entity_list}\n"
                f"相关论文：{', '.join(key_papers[:3])}"
            )
            try:
                response = await self.llm_client.chat.completions.create(
                    model=config.get("default_model") or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=512,
                )
                summary_text = response.choices[0].message.content.strip()
                self._update_community_summary(
                    comm_id,
                    summary=summary_text,
                    community_name=comm.get("semantic_tag", f"Community {comm_id}"),
                )
            except Exception as e:
                logger.warning(f"社区 {comm_id} 摘要生成失败: {e}")

        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            await asyncio.gather(*[_summarize_one(cid, c) for cid, c in batch])

        logger.info(f"社区摘要生成完成：{len(pending)} 个社区")
        return graph

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _infer_community_semantic_tag(
        self,
        dominant_types: list[str],
        labels: list[str],
    ) -> str:
        """启发式推断社区语义标签（在 LLM 摘要生成前作为 fallback）。"""
        type_set = set(dominant_types[:2])
        if "Paper" in type_set and "Method" in type_set:
            return "Method Research Cluster"
        if "Experiment" in type_set or "Result" in type_set:
            return "Experimental Results Cluster"
        if "Dataset" in type_set:
            return "Benchmark & Dataset Cluster"
        if "Contribution" in type_set:
            return "Research Contributions Cluster"
        if "Task" in type_set:
            return "Task & Application Cluster"
        if "Metric" in type_set:
            return "Evaluation Metrics Cluster"
        combined = " ".join(labels[:5]).lower()
        if any(k in combined for k in ["transformer", "attention", "bert", "gpt", "llm"]):
            return "Language Model Cluster"
        if any(k in combined for k in ["image", "vision", "cnn", "detection"]):
            return "Computer Vision Cluster"
        if any(k in combined for k in ["graph", "knowledge", "kg", "ontology"]):
            return "Knowledge Graph Cluster"
        return "Research Topic Cluster"

    def _update_community_summary(
        self,
        comm_id: str,
        summary: str,
        community_name: str = "",
        key_entities: list[str] | None = None,
    ) -> None:
        """将 LLM 生成的摘要写回图谱中对应社区。"""
        communities = self.graph.get("communities", {})
        if comm_id not in communities:
            return
        communities[comm_id]["summary"] = summary
        if community_name:
            communities[comm_id]["community_name"] = community_name
        if key_entities:
            communities[comm_id]["key_entities"] = key_entities
