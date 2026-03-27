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

import re
from collections import defaultdict
from typing import Any

from src.core.embedding import VectorEmbedder
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

# ============================================================
# Schema 常量（与 graph_store.py 保持一致）
# ============================================================

VALID_ENTITY_TYPES = {
    "Paper", "Method", "Model", "Task", "Dataset",
    "Metric", "Experiment", "Result", "Contribution", "Concept",
}

VALID_RELATION_TYPES = {
    "proposes", "improves", "uses", "evaluates_on",
    "compared_with", "achieves", "applied_to", "related_to",
    "has_experiment", "uses_dataset", "measures", "produces",
    "has_contribution", "solves", "cites", "extends",
}

_RELATION_WEIGHT: dict[str, float] = {
    "proposes":        1.0,
    "has_contribution":1.0,
    "solves":          0.95,
    "improves":        0.95,
    "evaluates_on":    0.9,
    "produces":        0.9,
    "measures":        0.88,
    "has_experiment":  0.85,
    "achieves":        0.85,
    "uses_dataset":    0.82,
    "uses":            0.8,
    "applied_to":      0.8,
    "cites":           0.75,
    "extends":         0.75,
    "compared_with":   0.7,
    "related_to":      0.5,
}

_TYPE_BOOST: dict[str, float] = {
    "Method":       1.25,
    "Model":        1.20,
    "Experiment":   1.15,
    "Result":       1.15,
    "Dataset":      1.10,
    "Metric":       1.10,
    "Contribution": 1.10,
    "Task":         1.05,
    "Concept":      1.00,
    "Paper":        0.90,
}

_ABBREVIATION_MAP: dict[str, str] = {
    "bert":        "bidirectional encoder representations from transformers",
    "gpt":         "generative pre-trained transformer",
    "gpt-2":       "generative pre-trained transformer 2",
    "gpt-3":       "generative pre-trained transformer 3",
    "gpt-4":       "generative pre-trained transformer 4",
    "gpt4":        "generative pre-trained transformer 4",
    "t5":          "text-to-text transfer transformer",
    "llm":         "large language model",
    "llms":        "large language models",
    "nlp":         "natural language processing",
    "cv":          "computer vision",
    "rl":          "reinforcement learning",
    "dl":          "deep learning",
    "ml":          "machine learning",
    "gan":         "generative adversarial network",
    "cnn":         "convolutional neural network",
    "rnn":         "recurrent neural network",
    "lstm":        "long short-term memory",
    "transformer": "transformer",
    "attention":   "attention mechanism",
    "rag":         "retrieval augmented generation",
    "graphrag":    "graph retrieval augmented generation",
    "kg":          "knowledge graph",
    "kgs":         "knowledge graphs",
    "qa":          "question answering",
    "mt":          "machine translation",
    "nmt":         "neural machine translation",
    "bleu":        "bilingual evaluation understudy",
    "rouge":       "recall oriented understudy for gisting evaluation",
    "f1":          "f1 score",
    "acc":         "accuracy",
    "sota":        "state of the art",
    "finetune":    "fine-tuning",
    "fine-tune":   "fine-tuning",
    "pretraining": "pre-training",
    "pre-training":"pre-training",
    "zero-shot":   "zero-shot learning",
    "few-shot":    "few-shot learning",
}

_MODEL_VARIANT_PATTERN = re.compile(
    r"[-_](base|large|small|medium|xl|xxl|uncased|cased|"
    r"v\d+[\.\d]*|\d+[bm]|instruct|chat|hf|en|zh|multilingual)$",
    re.IGNORECASE,
)


# ============================================================
# 纯工具函数（无状态，可独立使用）
# ============================================================

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower())


def normalize_text(text: str) -> str:
    text = re.sub(r"[\(\)\[\]\{\}\|/\\,;:\"'`~!@#$%^&*_+=<>?]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def entity_id(entity_type: str, text: str) -> str:
    return f"{entity_type.lower()}:{normalize_text(text)}"


def relation_weight(rel: str) -> float:
    return _RELATION_WEIGHT.get(rel, 0.5)


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


def strip_model_variant(name: str) -> str:
    stripped = _MODEL_VARIANT_PATTERN.sub("", name.strip())
    return stripped if stripped else name


def expand_abbreviation(text: str) -> str:
    return _ABBREVIATION_MAP.get(text.lower(), text)


def jaccard_sim(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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
        从 reading_agent 结构化输出构建初始知识图谱。

        Args:
            papers:           原始论文列表（含 paper_id / title / citations）。
            extracted_papers: reading_agent 提取的结构化数据（Pydantic 模型或 dict）。
            db_id:            图谱 ID；若提供则重置当前图谱为新空图。

        Returns:
            构建完成的图谱字典（in-place，同时存储于 self.graph）。
        """
        if db_id is not None:
            self.graph = empty_graph(db_id)

        graph = self.graph
        edge_seen: set[tuple[str, str, str]] = set()

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
