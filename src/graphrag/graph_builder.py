"""
graph_builder.py — 面向对象的知识图谱构建器

职责：
  - 从 reading_agent 结构化输出构建实体图（build_from_papers）
  - 合并 LLM 抽取的 KG 三元组（merge_triples）
  - 三阶段实体链接（_resolve_entity）
  - 所有新节点通过注入的 VectorEmbedder 生成真实语义向量

依赖注入：
  GraphBuilder(embedder, graph_data=None)
  — embedder   : VectorEmbedder 实例
  — graph_data : 已有图谱字典（可选，续写模式）
"""
from __future__ import annotations

import copy
import json
import os
import re
from typing import Any

from src.core.config import config
from src.core.embedding import VectorEmbedder
from src.graphrag.schema import (
    VALID_ENTITY_TYPES,
    VALID_RELATION_TYPES,
    tokenize,
    normalize_text,
    entity_id,
    relation_weight,
    expand_abbreviation,
    strip_model_variant,
    jaccard_sim,
)
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return (
            [x.strip() for x in value.split(",") if x.strip()]
            if "," in value
            else ([value.strip()] if value.strip() else [])
        )
    return []


def empty_graph(db_id: str) -> dict[str, Any]:
    return {
        "db_id":          db_id,
        "nodes":          {},
        "edges":          [],
        "paper_entities": {},
        "entity_aliases": {},
        "communities":    {},
        "citation_graph": {},
        "stats":          {},
    }


def update_stats(graph: dict[str, Any]) -> None:
    type_count: dict[str, int] = {}
    for node in graph["nodes"].values():
        t = node.get("type", "Unknown")
        type_count[t] = type_count.get(t, 0) + 1
    citation_edge_count = sum(
        1 for e in graph.get("edges", []) if e.get("type") == "cites"
    )
    graph["stats"] = {
        "node_count":          len(graph["nodes"]),
        "edge_count":          len(graph["edges"]),
        "node_type_count":     type_count,
        "paper_count":         len(graph["paper_entities"]),
        "community_count":     len(graph.get("communities", {})),
        "citation_edge_count": citation_edge_count,
    }


# ============================================================
# GraphBuilder
# ============================================================

class GraphBuilder:
    """
    知识图谱构建器。

    通过依赖注入接收 VectorEmbedder，在创建每个新节点时
    调用 embedder.get_embedding() 生成真实语义向量，替代
    原 graph_store.py 中基于 SHA256 的伪哈希嵌入。

    Args:
        embedder:   VectorEmbedder 实例，负责生成节点语义向量。
        graph_data: 可选，已有图谱字典（续写/合并模式）；
                    若为 None 则初始化一个仅含必要骨架的空图。
    """

    def __init__(
        self,
        embedder: VectorEmbedder,
        graph_data: dict[str, Any] | None = None,
    ) -> None:
        self.embedder = embedder
        if graph_data is not None:
            self.graph = graph_data
        else:
            # 占位空图（无 db_id，由调用方后续填入）
            self.graph = empty_graph("__pending__")

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def build_from_papers(
        self,
        papers: list[dict[str, Any]],
        extracted_papers: list[Any],
        db_id: str | None = None,
    ) -> dict[str, Any]:
        """
        从 reading_agent 结构化输出**增量**构建知识图谱。

        已有节点和边不会被清除——新论文的实体会通过三阶段实体链接
        合并到现有图谱中。

        Args:
            papers:           原始论文列表（含 paper_id / title / citations）。
            extracted_papers: reading_agent 提取的结构化数据（Pydantic 模型或 dict）。
            db_id:            图谱 ID；仅用于标记 graph["db_id"]，不会重置图谱。

        Returns:
            构建完成的图谱字典（in-place，同时存储于 self.graph）。
        """
        if db_id is not None:
            self.graph["db_id"] = db_id

        graph = self.graph
        edge_seen: set[tuple[str, str, str]] = {
            (e["source"], e["target"], e["type"]) for e in graph.get("edges", [])
        }

        def add_node(node_id: str, node_type: str, label: str) -> None:
            if node_id not in graph["nodes"]:
                graph["nodes"][node_id] = {
                    "id":         node_id,
                    "type":       node_type,
                    "label":      label,
                    "norm_label": normalize_text(label),
                    "embedding":  self.embedder.get_embedding(f"{node_type}: {label}"),
                }
            graph["entity_aliases"][graph["nodes"][node_id]["norm_label"]] = node_id

        def resolve_or_add(raw_name: str, etype: str) -> str:
            existing = self._resolve_entity(raw_name, etype)
            if existing:
                return existing
            nid = entity_id(etype, raw_name)
            add_node(nid, etype, raw_name)
            return nid

        def add_edge(src: str, dst: str, rel: str, metadata: dict | None = None) -> None:
            key = (src, dst, rel)
            if key in edge_seen or src == dst:
                return
            edge_seen.add(key)
            entry: dict[str, Any] = {
                "source": src,
                "target": dst,
                "type":   rel,
                "weight": relation_weight(rel),
            }
            if metadata:
                entry["metadata"] = metadata
            graph["edges"].append(entry)

        for idx, paper in enumerate(papers):
            paper_id_raw = str(paper.get("paper_id") or paper.get("id") or f"paper_{idx}")
            paper_node_id = entity_id("Paper", paper_id_raw)
            title = str(paper.get("title", paper_id_raw))
            add_node(paper_node_id, "Paper", title)
            graph["paper_entities"][paper_id_raw] = [paper_node_id]

            citations = safe_list(paper.get("citations") or paper.get("references"))
            if citations:
                graph["citation_graph"][paper_id_raw] = citations

            extracted = extracted_papers[idx] if idx < len(extracted_papers) else None
            if extracted is None:
                continue

            ext: dict[str, Any] = (
                extracted.model_dump() if hasattr(extracted, "model_dump") else
                extracted if isinstance(extracted, dict) else {}
            )

            # Method / Model
            method_name = str((ext.get("key_methodology") or {}).get("name") or "").strip()
            method_node_id: str | None = None
            if method_name:
                method_node_id = resolve_or_add(method_name, "Method")
                add_edge(paper_node_id, method_node_id, "proposes")
                graph["paper_entities"][paper_id_raw].append(method_node_id)

            # Task
            core_problem = str(ext.get("core_problem") or "").strip()
            task_node_id: str | None = None
            if core_problem:
                task_node_id = resolve_or_add(core_problem[:100], "Task")
                add_edge(paper_node_id, task_node_id, "applied_to")
                if method_node_id:
                    add_edge(method_node_id, task_node_id, "solves")
                graph["paper_entities"][paper_id_raw].append(task_node_id)

            # Dataset
            for ds in safe_list(ext.get("datasets_used")):
                ds_node_id = resolve_or_add(ds, "Dataset")
                add_edge(paper_node_id, ds_node_id, "uses")
                if method_node_id:
                    add_edge(method_node_id, ds_node_id, "evaluates_on")
                graph["paper_entities"][paper_id_raw].append(ds_node_id)

            # Metric
            for metric in safe_list(ext.get("evaluation_metrics")):
                met_node_id = resolve_or_add(metric, "Metric")
                if method_node_id:
                    add_edge(method_node_id, met_node_id, "achieves")
                else:
                    add_edge(paper_node_id, met_node_id, "achieves")
                graph["paper_entities"][paper_id_raw].append(met_node_id)

            # Experiment + Result
            experiments = safe_list(ext.get("experiments") or ext.get("experiment_results"))
            for exp_text in experiments[:5]:
                exp_label = exp_text[:150]
                exp_node_id = resolve_or_add(exp_label, "Experiment")
                add_edge(paper_node_id, exp_node_id, "has_experiment")

                result_matches = re.findall(
                    r"(\w[\w\s\-]+?)\s*(?:=|:)\s*(\d+\.?\d*\s*%?)",
                    exp_text,
                )
                for metric_name, value in result_matches[:3]:
                    result_label = f"{metric_name.strip()}: {value.strip()}"
                    result_node_id = resolve_or_add(result_label, "Result")
                    add_edge(exp_node_id, result_node_id, "produces")
                    for ds in safe_list(ext.get("datasets_used"))[:2]:
                        ds_nid = entity_id("Dataset", ds)
                        if ds_nid in graph["nodes"]:
                            add_edge(exp_node_id, ds_nid, "uses_dataset")

                graph["paper_entities"][paper_id_raw].append(exp_node_id)

            # Contribution
            for contrib in safe_list(ext.get("contributions"))[:4]:
                contrib_label = contrib[:150]
                contrib_node_id = resolve_or_add(contrib_label, "Contribution")
                add_edge(paper_node_id, contrib_node_id, "has_contribution")
                if method_node_id:
                    add_edge(method_node_id, contrib_node_id, "related_to")
                graph["paper_entities"][paper_id_raw].append(contrib_node_id)

        self._build_citation_edges(edge_seen)
        update_stats(graph)
        logger.info(
            f"图谱构建完成：{graph['stats'].get('node_count', 0)} 节点，"
            f"{graph['stats'].get('edge_count', 0)} 边"
        )
        return graph

    def merge_triples(
        self,
        kg_output: dict[str, Any],
        paper_id_raw: str | None = None,
    ) -> dict[str, Any]:
        """
        将 kg_extraction_prompt 输出的 JSON（entities + relations）
        合并进 self.graph（三阶段实体链接 + 真实 Embedding 生成）。

        Args:
            kg_output:    LLM 输出的 {"entities": [...], "relations": [...]}
            paper_id_raw: 可选，关联到哪篇论文

        Returns:
            更新后的图谱（in-place）
        """
        graph = self.graph
        if not isinstance(kg_output, dict):
            return graph

        local_id_map: dict[str, str] = {}

        for ent in kg_output.get("entities", []):
            raw_name = str(ent.get("name", "")).strip()
            raw_type = str(ent.get("type", "Concept")).strip()
            local_id = str(ent.get("id", "")).strip()
            desc     = str(ent.get("description", "")).strip()

            if not raw_name:
                continue
            if raw_type not in VALID_ENTITY_TYPES:
                raw_type = "Concept"

            existing = self._resolve_entity(raw_name, raw_type)
            if existing:
                canonical_id = existing
                if desc and not graph["nodes"][canonical_id].get("description"):
                    graph["nodes"][canonical_id]["description"] = desc
            else:
                canonical_id = entity_id(raw_type, raw_name)
                norm = normalize_text(raw_name)
                graph["nodes"][canonical_id] = {
                    "id":          canonical_id,
                    "type":        raw_type,
                    "label":       raw_name,
                    "norm_label":  norm,
                    "description": desc,
                    "embedding":   self.embedder.get_embedding(f"{raw_type}: {raw_name}"),
                }
                graph["entity_aliases"][norm] = canonical_id
                expanded = expand_abbreviation(norm)
                if expanded != norm:
                    graph["entity_aliases"].setdefault(expanded, canonical_id)

            local_id_map[local_id] = canonical_id

            if paper_id_raw:
                lst = graph["paper_entities"].setdefault(paper_id_raw, [])
                if canonical_id not in lst:
                    lst.append(canonical_id)

        edge_seen = {(e["source"], e["target"], e["type"]) for e in graph["edges"]}

        for rel in kg_output.get("relations", []):
            src_local = str(rel.get("source", "")).strip()
            dst_local = str(rel.get("target", "")).strip()
            rel_type  = str(rel.get("relation", "related_to")).strip()
            rel_meta  = rel.get("metadata")

            if rel_type not in VALID_RELATION_TYPES:
                rel_type = "related_to"

            src_id = local_id_map.get(src_local)
            dst_id = local_id_map.get(dst_local)

            if not src_id or not dst_id or src_id == dst_id:
                continue

            key = (src_id, dst_id, rel_type)
            if key in edge_seen:
                continue
            edge_seen.add(key)

            entry: dict[str, Any] = {
                "source": src_id,
                "target": dst_id,
                "type":   rel_type,
                "weight": relation_weight(rel_type),
            }
            if rel_meta:
                entry["metadata"] = rel_meta
            graph["edges"].append(entry)

        update_stats(graph)
        return graph

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _resolve_entity(
        self,
        raw_name: str,
        entity_type: str,
        fuzzy_threshold: float = 0.72,
    ) -> str | None:
        """
        三阶段实体链接：
          Stage 1: 精确 norm_label 匹配（entity_aliases）
          Stage 2: 缩写展开后精确匹配 + 变体后缀剥离匹配
          Stage 3: Token Jaccard Fuzzy 匹配（阈值 ≥ fuzzy_threshold）

        命中返回 canonical_id；未命中返回 None。
        """
        graph = self.graph
        aliases = graph.get("entity_aliases", {})
        nodes   = graph.get("nodes", {})

        # Stage 1: 精确归一化匹配
        norm = normalize_text(raw_name)
        if norm in aliases:
            return aliases[norm]

        # Stage 2a: 缩写展开
        expanded = expand_abbreviation(norm)
        if expanded != norm and expanded in aliases:
            return aliases[expanded]

        # Stage 2b: 变体后缀剥离
        stripped = normalize_text(strip_model_variant(raw_name))
        if stripped != norm and stripped in aliases:
            return aliases[stripped]

        # Stage 3: Fuzzy Jaccard（仅同类型节点）
        query_tokens = set(tokenize(norm))
        if not query_tokens:
            return None

        best_id: str | None = None
        best_sim = fuzzy_threshold
        for node_id, node in nodes.items():
            if node.get("type") != entity_type:
                continue
            cand_tokens = set(tokenize(node.get("norm_label", "")))
            sim = jaccard_sim(query_tokens, cand_tokens)
            if sim > best_sim:
                best_sim = sim
                best_id = node_id

        return best_id

    def _build_citation_edges(
        self,
        edge_seen: set[tuple[str, str, str]],
    ) -> None:
        """根据 citation_graph 在图谱中添加 Paper → cites → Paper 边。"""
        graph    = self.graph
        citation_graph = graph.get("citation_graph", {})
        nodes    = graph.get("nodes", {})
        aliases  = graph.get("entity_aliases", {})

        for src_pid, cited_list in citation_graph.items():
            src_node_id = entity_id("Paper", src_pid)
            if src_node_id not in nodes:
                continue
            for cited in cited_list:
                norm_cited  = normalize_text(str(cited))
                dst_node_id = aliases.get(norm_cited)
                if not dst_node_id:
                    dst_node_id = entity_id("Paper", str(cited))
                    if dst_node_id not in nodes:
                        continue
                key = (src_node_id, dst_node_id, "cites")
                if key not in edge_seen and src_node_id != dst_node_id:
                    edge_seen.add(key)
                    graph["edges"].append({
                        "source": src_node_id,
                        "target": dst_node_id,
                        "type":   "cites",
                        "weight": relation_weight("cites"),
                    })


# ============================================================
# 图谱持久化工具函数
# ============================================================

def _graph_dir() -> str:
    root = os.path.join(config.get("SAVE_DIR", "data"), "knowledge_graphs")
    os.makedirs(root, exist_ok=True)
    return root


# ------------------------------------------------------------------
# 图谱内存缓存（按 db_id + 文件 mtime 键控）
# ------------------------------------------------------------------
_graph_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_GRAPH_CACHE_MAX = 20


def _evict_graph_cache() -> None:
    """淘汰最旧的缓存条目直至不超过上限。"""
    while len(_graph_cache) > _GRAPH_CACHE_MAX:
        oldest_key = min(_graph_cache, key=lambda k: _graph_cache[k][0])
        _graph_cache.pop(oldest_key, None)


def invalidate_graph_cache(db_id: str) -> None:
    """保存图谱后主动失效对应缓存条目。"""
    _graph_cache.pop(db_id, None)


def save_entity_graph(db_id: str, graph: dict[str, Any]) -> str:
    """将图谱写入 JSON 文件，保存前剥除节点 embedding 以减少磁盘占用。"""
    g = copy.deepcopy(graph)
    for node in g.get("nodes", {}).values():
        node.pop("embedding", None)
    path = os.path.join(_graph_dir(), f"{db_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False, indent=2)

    invalidate_graph_cache(db_id)

    logger.info(
        f"图谱已保存：{path}（{graph.get('stats', {}).get('node_count', 0)} 节点，"
        f"{graph.get('stats', {}).get('edge_count', 0)} 边）"
    )
    return path


def load_entity_graph(
    db_id: str,
    embedder: VectorEmbedder | None = None,
) -> dict[str, Any] | None:
    """
    从磁盘加载图谱（带内存缓存）。

    同一 db_id 在文件未变更时直接返回缓存的深拷贝（写时复制安全）。
    embedder 仅在首次加载或缓存失效时用于重建节点语义向量。
    """
    path = os.path.join(_graph_dir(), f"{db_id}.json")
    if not os.path.exists(path):
        return None

    try:
        file_mtime = os.path.getmtime(path)
    except OSError:
        return None

    cached = _graph_cache.get(db_id)
    if cached is not None and cached[0] >= file_mtime:
        return copy.deepcopy(cached[1])

    try:
        with open(path, encoding="utf-8") as f:
            graph = json.load(f)
        if embedder is not None:
            for node in graph.get("nodes", {}).values():
                if "embedding" not in node:
                    node["embedding"] = embedder.get_embedding(
                        f"{node.get('type', 'Concept')}: {node.get('label', '')}"
                    )

        _graph_cache[db_id] = (file_mtime, copy.deepcopy(graph))
        _evict_graph_cache()

        return graph
    except Exception as e:
        logger.warning(f"加载图谱失败 {db_id}: {e}")
        return None


def graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    """返回图谱统计信息（node_count / edge_count / paper_count / node_type_count）。"""
    if not graph:
        return {"node_count": 0, "edge_count": 0, "paper_count": 0, "node_type_count": {}}
    return graph.get("stats", {})
