"""
graph_store.py — 知识图谱存储与 GraphRAG 核心服务 (v2)

Schema（实体类型）:
    Paper, Method, Model, Task, Dataset, Metric,
    Experiment, Result, Contribution, Concept

关系类型（扩展）:
    proposes, improves, uses, evaluates_on, compared_with,
    achieves, applied_to, related_to,
    has_experiment, uses_dataset, measures, produces,
    has_contribution, solves, cites, extends

设计目标：
    1. 研究级 GraphRAG 架构 — 三层检索（Local / Community / Global）
    2. 实体归一化 + 三阶段链接（精确 → 缩写展开 → Fuzzy）
    3. 图节点语义向量（基于类型+标签，可换 sentence-transformer 扩展）
    4. 五分量融合评分（vec / graph / entity / embedding / lexical）
    5. 结构化局部子图语义化（知识三元组 → 可读知识块）
    6. 引用图（Paper → cites → Paper）+ 引用链推理
    7. 异步社区摘要批量生成（LLM）
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from typing import Any

from src.core.config import config
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:
    GraphDatabase = None  # type: ignore


# ============================================================
# Schema 常量（v2 扩展）
# ============================================================

VALID_ENTITY_TYPES = {
    "Paper", "Method", "Model", "Task", "Dataset",
    "Metric", "Experiment", "Result", "Contribution", "Concept",
}

VALID_RELATION_TYPES = {
    # 原有
    "proposes", "improves", "uses", "evaluates_on",
    "compared_with", "achieves", "applied_to", "related_to",
    # 新增 — 实验/结果/贡献
    "has_experiment", "uses_dataset", "measures", "produces",
    "has_contribution", "solves",
    # 新增 — 引用图
    "cites", "extends",
}

# 关系语义化标签（用于 subgraph 文本渲染）
RELATION_LABELS: dict[str, str] = {
    "proposes":        "proposes",
    "improves":        "improves upon",
    "uses":            "uses",
    "evaluates_on":    "evaluated on",
    "compared_with":   "compared with",
    "achieves":        "achieves",
    "applied_to":      "applied to",
    "related_to":      "related to",
    "has_experiment":  "has experiment",
    "uses_dataset":    "uses dataset",
    "measures":        "measures",
    "produces":        "produces result",
    "has_contribution":"contributes",
    "solves":          "solves",
    "cites":           "cites",
    "extends":         "extends",
}

# 关系权重（图传播强度）
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

# 实体类型在图传播中的重要性权重
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

# 缩写字典（可扩展）— 用于实体链接第二阶段
_ABBREVIATION_MAP: dict[str, str] = {
    "bert":            "bidirectional encoder representations from transformers",
    "gpt":             "generative pre-trained transformer",
    "gpt-2":           "generative pre-trained transformer 2",
    "gpt-3":           "generative pre-trained transformer 3",
    "gpt-4":           "generative pre-trained transformer 4",
    "gpt4":            "generative pre-trained transformer 4",
    "t5":              "text-to-text transfer transformer",
    "llm":             "large language model",
    "llms":            "large language models",
    "nlp":             "natural language processing",
    "cv":              "computer vision",
    "rl":              "reinforcement learning",
    "dl":              "deep learning",
    "ml":              "machine learning",
    "gan":             "generative adversarial network",
    "cnn":             "convolutional neural network",
    "rnn":             "recurrent neural network",
    "lstm":            "long short-term memory",
    "transformer":     "transformer",
    "attention":       "attention mechanism",
    "rag":             "retrieval augmented generation",
    "graphrag":        "graph retrieval augmented generation",
    "kg":              "knowledge graph",
    "kgs":             "knowledge graphs",
    "qa":              "question answering",
    "mt":              "machine translation",
    "nmt":             "neural machine translation",
    "bleu":            "bilingual evaluation understudy",
    "rouge":           "recall oriented understudy for gisting evaluation",
    "f1":              "f1 score",
    "acc":             "accuracy",
    "sota":            "state of the art",
    "finetune":        "fine-tuning",
    "fine-tune":       "fine-tuning",
    "pretraining":     "pre-training",
    "pre-training":    "pre-training",
    "zero-shot":       "zero-shot learning",
    "few-shot":        "few-shot learning",
}

# 模型变体后缀（用于归一化剥离）— BERT-base-uncased → BERT
_MODEL_VARIANT_PATTERN = re.compile(
    r"[-_](base|large|small|medium|xl|xxl|uncased|cased|"
    r"v\d+[\.\d]*|\d+[bm]|instruct|chat|hf|en|zh|multilingual)$",
    re.IGNORECASE,
)


# ============================================================
# 工具函数
# ============================================================

def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower())


def _normalize_text(text: str) -> str:
    """去除标点、多余空白，转小写，用于实体归一化键"""
    text = re.sub(r"[\(\)\[\]\{\}\|/\\,;:\"'`~!@#$%^&*_+=<>?]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _entity_id(entity_type: str, text: str) -> str:
    return f"{entity_type.lower()}:{_normalize_text(text)}"


def _relation_weight(rel: str) -> float:
    return _RELATION_WEIGHT.get(rel, 0.5)


def _type_boost(node_type: str) -> float:
    return _TYPE_BOOST.get(node_type, 1.0)


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return ([x.strip() for x in value.split(",") if x.strip()]
                if "," in value else ([value.strip()] if value.strip() else []))
    return []


def _strip_model_variant(name: str) -> str:
    """剥离模型变体后缀：BERT-base-uncased → BERT"""
    stripped = _MODEL_VARIANT_PATTERN.sub("", name.strip())
    return stripped if stripped else name


def _expand_abbreviation(text: str) -> str:
    """将缩写展开为规范名称（小写查找）"""
    return _ABBREVIATION_MAP.get(text.lower(), text)


def _jaccard_sim(a: set[str], b: set[str]) -> float:
    """Jaccard 相似度，用于 fuzzy 实体匹配"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ============================================================
# 实体链接 — 三阶段归一化
#   Stage 1: 精确 norm_label 匹配（已存在的 entity_aliases）
#   Stage 2: 缩写展开后再精确匹配
#   Stage 3: Token Jaccard Fuzzy 匹配（相似度 ≥ 0.7）
# ============================================================

def _resolve_entity(
    raw_name: str,
    graph: dict[str, Any],
    entity_type: str,
    fuzzy_threshold: float = 0.72,
) -> str | None:
    """
    三阶段实体链接，返回已存在节点的 canonical_id；
    未命中返回 None（调用方创建新节点）。
    """
    aliases = graph.get("entity_aliases", {})
    nodes = graph.get("nodes", {})

    # Stage 1: 精确归一化匹配
    norm = _normalize_text(raw_name)
    if norm in aliases:
        return aliases[norm]

    # Stage 2: 缩写展开
    expanded = _expand_abbreviation(norm)
    if expanded != norm and expanded in aliases:
        return aliases[expanded]

    # 剥离变体后缀再匹配
    stripped = _normalize_text(_strip_model_variant(raw_name))
    if stripped != norm and stripped in aliases:
        return aliases[stripped]

    # Stage 3: Fuzzy — 仅对同类型节点做 Jaccard
    query_tokens = set(_tokenize(norm))
    if not query_tokens:
        return None

    best_id: str | None = None
    best_sim = fuzzy_threshold
    for node_id, node in nodes.items():
        if node.get("type") != entity_type:
            continue
        cand_tokens = set(_tokenize(node.get("norm_label", "")))
        sim = _jaccard_sim(query_tokens, cand_tokens)
        if sim > best_sim:
            best_sim = sim
            best_id = node_id

    return best_id


# ============================================================
# 图节点语义向量（伪嵌入）
# 方案：hash 类型+标签 → 确定性单位向量（dim=64）
# 后续可替换为 sentence-transformers 编码；接口不变。
# ============================================================

_EMBED_DIM = 64


def _compute_node_embedding(node_type: str, label: str) -> list[float]:
    """
    基于节点类型 + 标签计算伪嵌入向量（可替换为真实模型）。
    - 类型决定"方向基调"，标签决定"细粒度偏移"。
    - 使用 SHA256 哈希保证确定性，归一化到单位球。
    """
    text = f"{node_type}:{label}"
    digest = hashlib.sha256(text.encode()).digest()  # 32 bytes
    # 将 64 bytes（通过两次 hash）映射到 64 维
    digest2 = hashlib.sha256((text + "_v2").encode()).digest()
    raw = list(digest) + list(digest2)  # 64 ints in [0, 255]
    # 中心化到 [-1, 1]
    vec = [(v / 127.5 - 1.0) for v in raw]
    # 加入类型偏置（让同类型节点聚类）
    type_bias = _TYPE_BOOST.get(node_type, 1.0) - 1.0
    for i in range(0, min(8, _EMBED_DIM)):
        vec[i] += type_bias * 0.5
    # 归一化
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _embedding_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度，向量已归一化，直接点积"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, (dot + 1.0) / 2.0))  # 映射到 [0,1]


def _get_query_embedding(query_text: str) -> list[float]:
    """查询文本的向量：对 Concept 类型节点做近似"""
    return _compute_node_embedding("Concept", _normalize_text(query_text)[:200])


# ============================================================
# 图谱文件路径
# ============================================================

def _graph_dir() -> str:
    root = os.path.join(config.get("SAVE_DIR"), "knowledge_graphs")
    os.makedirs(root, exist_ok=True)
    return root


def _graph_path(db_id: str) -> str:
    return os.path.join(_graph_dir(), f"{db_id}.json")


# ============================================================
# 空图初始化
# ============================================================

def _empty_graph(db_id: str) -> dict[str, Any]:
    return {
        "db_id":           db_id,
        "nodes":           {},
        "edges":           [],
        "paper_entities":  {},
        "entity_aliases":  {},
        "communities":     {},
        "citation_graph":  {},   # paper_id_raw → [cited_paper_id_raw, ...]
        "stats":           {},
    }


# ============================================================
# 图谱构建 — 从 reading_agent 结构化输出
# ============================================================

def build_entity_graph_from_papers(
    papers: list[dict[str, Any]],
    extracted_papers: list[Any],
    db_id: str,
) -> dict[str, Any]:
    """
    从 reading_agent 提取的结构化论文数据构建初始知识图谱（v2）。

    新增：
    - Experiment 节点（每篇论文的主要实验）
    - Result 节点（具体数值/结论）
    - Contribution 节点（贡献点）
    - 三阶段实体链接（归一化去重）
    - 节点 embedding 存储
    - 引用图初始化（citations 字段）
    """
    graph = _empty_graph(db_id)
    edge_seen: set[tuple[str, str, str]] = set()

    def add_node(node_id: str, node_type: str, label: str) -> None:
        if node_id not in graph["nodes"]:
            graph["nodes"][node_id] = {
                "id":         node_id,
                "type":       node_type,
                "label":      label,
                "norm_label": _normalize_text(label),
                "embedding":  _compute_node_embedding(node_type, label),
            }
        graph["entity_aliases"][graph["nodes"][node_id]["norm_label"]] = node_id

    def resolve_or_add(raw_name: str, entity_type: str) -> str:
        """实体链接：先三阶段查找，未命中则创建新节点"""
        existing = _resolve_entity(raw_name, graph, entity_type)
        if existing:
            return existing
        nid = _entity_id(entity_type, raw_name)
        add_node(nid, entity_type, raw_name)
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
            "weight": _relation_weight(rel),
        }
        if metadata:
            entry["metadata"] = metadata
        graph["edges"].append(entry)

    for idx, paper in enumerate(papers):
        paper_id_raw = str(paper.get("paper_id") or paper.get("id") or f"paper_{idx}")
        paper_node_id = _entity_id("Paper", paper_id_raw)
        title = str(paper.get("title", paper_id_raw))
        add_node(paper_node_id, "Paper", title)
        graph["paper_entities"][paper_id_raw] = [paper_node_id]

        # ---- 引用图初始化（citations 字段） ----
        citations = _safe_list(paper.get("citations") or paper.get("references"))
        if citations:
            graph["citation_graph"][paper_id_raw] = citations

        extracted = extracted_papers[idx] if idx < len(extracted_papers) else None
        if extracted is None:
            continue

        ext: dict[str, Any] = (
            extracted.model_dump() if hasattr(extracted, "model_dump") else
            extracted if isinstance(extracted, dict) else {}
        )

        # ---- Method / Model ----
        method_name = str((ext.get("key_methodology") or {}).get("name") or "").strip()
        method_node_id: str | None = None
        if method_name:
            method_node_id = resolve_or_add(method_name, "Method")
            add_edge(paper_node_id, method_node_id, "proposes")
            graph["paper_entities"][paper_id_raw].append(method_node_id)

        # ---- Task（core_problem） ----
        core_problem = str(ext.get("core_problem") or "").strip()
        task_node_id: str | None = None
        if core_problem:
            task_node_id = resolve_or_add(core_problem[:100], "Task")
            add_edge(paper_node_id, task_node_id, "applied_to")
            if method_node_id:
                add_edge(method_node_id, task_node_id, "solves")
            graph["paper_entities"][paper_id_raw].append(task_node_id)

        # ---- Dataset ----
        for ds in _safe_list(ext.get("datasets_used")):
            ds_node_id = resolve_or_add(ds, "Dataset")
            add_edge(paper_node_id, ds_node_id, "uses")
            if method_node_id:
                add_edge(method_node_id, ds_node_id, "evaluates_on")
            graph["paper_entities"][paper_id_raw].append(ds_node_id)

        # ---- Metric ----
        for metric in _safe_list(ext.get("evaluation_metrics")):
            met_node_id = resolve_or_add(metric, "Metric")
            if method_node_id:
                add_edge(method_node_id, met_node_id, "achieves")
            else:
                add_edge(paper_node_id, met_node_id, "achieves")
            graph["paper_entities"][paper_id_raw].append(met_node_id)

        # ---- Experiment + Result（新增） ----
        experiments = _safe_list(ext.get("experiments") or ext.get("experiment_results"))
        for exp_text in experiments[:5]:  # 最多5个实验
            exp_label = exp_text[:150]
            exp_node_id = resolve_or_add(exp_label, "Experiment")
            add_edge(paper_node_id, exp_node_id, "has_experiment")

            # 从实验文本中提取数值结果（简单规则）
            result_matches = re.findall(
                r"(\w[\w\s\-]+?)\s*(?:=|:)\s*(\d+\.?\d*\s*%?)",
                exp_text,
            )
            for metric_name, value in result_matches[:3]:
                result_label = f"{metric_name.strip()}: {value.strip()}"
                result_node_id = resolve_or_add(result_label, "Result")
                add_edge(exp_node_id, result_node_id, "produces")
                for ds in _safe_list(ext.get("datasets_used"))[:2]:
                    ds_node_id = _entity_id("Dataset", ds)
                    if ds_node_id in graph["nodes"]:
                        add_edge(exp_node_id, ds_node_id, "uses_dataset")

            graph["paper_entities"][paper_id_raw].append(exp_node_id)

        # ---- Contribution（新增） ----
        contributions = _safe_list(ext.get("contributions"))
        for contrib in contributions[:4]:
            contrib_label = contrib[:150]
            contrib_node_id = resolve_or_add(contrib_label, "Contribution")
            add_edge(paper_node_id, contrib_node_id, "has_contribution")
            if method_node_id:
                add_edge(method_node_id, contrib_node_id, "related_to")
            graph["paper_entities"][paper_id_raw].append(contrib_node_id)

    # ---- 构建跨论文引用边 ----
    _build_citation_edges(graph, edge_seen)

    _update_stats(graph)
    return graph


def _build_citation_edges(
    graph: dict[str, Any],
    edge_seen: set[tuple[str, str, str]],
) -> None:
    """
    根据 citation_graph 在图谱中添加 Paper → cites → Paper 边。
    引用目标若尚未在图谱中，则跳过（不创建孤立节点）。
    """
    citation_graph = graph.get("citation_graph", {})
    nodes = graph.get("nodes", {})
    aliases = graph.get("entity_aliases", {})

    for src_pid, cited_list in citation_graph.items():
        src_node_id = _entity_id("Paper", src_pid)
        if src_node_id not in nodes:
            continue
        for cited in cited_list:
            # 尝试通过归一化标题匹配目标节点
            norm_cited = _normalize_text(str(cited))
            dst_node_id = aliases.get(norm_cited)
            if not dst_node_id:
                # 尝试通过 paper_id 匹配
                dst_node_id = _entity_id("Paper", str(cited))
                if dst_node_id not in nodes:
                    continue
            key = (src_node_id, dst_node_id, "cites")
            if key not in edge_seen and src_node_id != dst_node_id:
                edge_seen.add(key)
                graph["edges"].append({
                    "source": src_node_id,
                    "target": dst_node_id,
                    "type":   "cites",
                    "weight": _relation_weight("cites"),
                })


# ============================================================
# 合并 LLM 抽取的三元组（KG Extraction Prompt 输出）
# ============================================================

def merge_kg_triples(
    graph: dict[str, Any],
    kg_output: dict[str, Any],
    paper_id_raw: str | None = None,
) -> dict[str, Any]:
    """
    将 kg_extraction_prompt 输出的 JSON（entities + relations）合并进现有图谱。
    使用三阶段实体链接 + 自动 embedding 生成。

    Args:
        graph:        现有图谱（in-place 修改）
        kg_output:    LLM 输出的 {"entities": [...], "relations": [...]}
        paper_id_raw: 可选，关联到哪篇论文
    """
    if not isinstance(kg_output, dict):
        return graph

    local_id_map: dict[str, str] = {}

    for ent in kg_output.get("entities", []):
        raw_name  = str(ent.get("name", "")).strip()
        raw_type  = str(ent.get("type", "Concept")).strip()
        local_id  = str(ent.get("id", "")).strip()
        desc      = str(ent.get("description", "")).strip()

        if not raw_name:
            continue
        if raw_type not in VALID_ENTITY_TYPES:
            raw_type = "Concept"

        # 三阶段实体链接
        existing = _resolve_entity(raw_name, graph, raw_type)
        if existing:
            canonical_id = existing
            # 追加描述（若原节点没有）
            if desc and not graph["nodes"][canonical_id].get("description"):
                graph["nodes"][canonical_id]["description"] = desc
        else:
            canonical_id = _entity_id(raw_type, raw_name)
            norm = _normalize_text(raw_name)
            graph["nodes"][canonical_id] = {
                "id":          canonical_id,
                "type":        raw_type,
                "label":       raw_name,
                "norm_label":  norm,
                "description": desc,
                "embedding":   _compute_node_embedding(raw_type, raw_name),
            }
            graph["entity_aliases"][norm] = canonical_id
            # 缩写展开别名
            expanded = _expand_abbreviation(norm)
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
        rel_meta  = rel.get("metadata")  # 可选附加属性，如 value

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
            "weight": _relation_weight(rel_type),
        }
        if rel_meta:
            entry["metadata"] = rel_meta
        graph["edges"].append(entry)

    _update_stats(graph)
    return graph


# ============================================================
# 社区聚类 — Louvain（纯 Python 实现）
# ============================================================

def build_communities(graph: dict[str, Any]) -> dict[str, Any]:
    """
    Louvain 社区聚类（v2）。
    额外存储：
    - dominant_types：社区主导实体类型
    - key_papers：社区内的 Paper 节点标签
    - semantic_tags：由主导类型+关键实体推断的语义标签
    """
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
                nbr_comms = [community[nbr] for nbr in adj.get(node_id, {}) if community[nbr] != scid]
                target = nbr_comms[0] if nbr_comms else largest_comm
                comm_nodes[target].append(node_id)
                del comm_nodes[scid]

    communities: dict[str, dict] = {}
    node_data = graph["nodes"]
    for new_id, (_, node_ids) in enumerate(comm_nodes.items()):
        labels    = [node_data[nid]["label"] for nid in node_ids if nid in node_data]
        types     = [node_data[nid]["type"]  for nid in node_ids if nid in node_data]
        key_papers = [
            node_data[nid]["label"]
            for nid in node_ids
            if nid in node_data and node_data[nid]["type"] == "Paper"
        ]

        # 主导类型统计
        type_counter: dict[str, int] = {}
        for t in types:
            type_counter[t] = type_counter.get(t, 0) + 1
        dominant_types = sorted(type_counter, key=lambda x: -type_counter[x])[:3]

        # 语义标签（启发式）
        semantic_tag = _infer_community_semantic_tag(dominant_types, labels)

        communities[str(new_id)] = {
            "nodes":          node_ids,
            "key_entities":   labels[:12],
            "key_papers":     key_papers[:5],
            "dominant_types": dominant_types,
            "semantic_tag":   semantic_tag,
            "summary":        "",   # 由 LLM 后续异步填充
            "community_name": semantic_tag,
        }

    graph["communities"] = communities
    logger.info(f"Louvain 聚类完成：{len(communities)} 个社区，{len(nodes)} 个节点")
    return graph


def _infer_community_semantic_tag(dominant_types: list[str], labels: list[str]) -> str:
    """启发式推断社区语义标签（在 LLM 摘要生成前使用）"""
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
    # 尝试从标签提取语义
    combined = " ".join(labels[:5]).lower()
    if any(k in combined for k in ["transformer", "attention", "bert", "gpt", "llm"]):
        return "Language Model Cluster"
    if any(k in combined for k in ["image", "vision", "cnn", "detection"]):
        return "Computer Vision Cluster"
    if any(k in combined for k in ["graph", "knowledge", "kg", "ontology"]):
        return "Knowledge Graph Cluster"
    return "Research Topic Cluster"


def update_community_summary(
    graph: dict[str, Any],
    comm_id: str,
    summary: str,
    community_name: str = "",
    key_entities: list[str] | None = None,
) -> None:
    """将 LLM 生成的社区摘要写回图谱"""
    if "communities" not in graph or comm_id not in graph["communities"]:
        return
    graph["communities"][comm_id]["summary"] = summary
    if community_name:
        graph["communities"][comm_id]["community_name"] = community_name
    if key_entities:
        graph["communities"][comm_id]["key_entities"] = key_entities


async def generate_all_community_summaries(
    graph: dict[str, Any],
    llm_client: Any,
    batch_size: int = 4,
) -> dict[str, Any]:
    """
    异步批量为所有无摘要社区调用 LLM 生成语义摘要。

    Args:
        graph:      已建立社区的图谱
        llm_client: 支持 async create 的 LLM 客户端
        batch_size: 并发批次大小

    Returns:
        更新了社区摘要的图谱（in-place）
    """
    from src.core.prompts import community_summary_prompt

    communities = graph.get("communities", {})
    pending = [
        (comm_id, comm)
        for comm_id, comm in communities.items()
        if not comm.get("summary")
    ]

    if not pending:
        return graph

    async def _summarize_one(comm_id: str, comm: dict) -> None:
        key_entities = comm.get("key_entities", [])
        key_papers   = comm.get("key_papers", [])
        dominant_types = comm.get("dominant_types", [])
        entity_list  = "\n".join(f"- {e}" for e in key_entities[:15])
        prompt = (
            community_summary_prompt
            + f"\n\n【社区信息】\n"
            f"主导实体类型：{', '.join(dominant_types)}\n"
            f"关键实体：\n{entity_list}\n"
            f"相关论文：{', '.join(key_papers[:3])}"
        )
        try:
            response = await llm_client.chat.completions.create(
                model=config.get("default_model") or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            summary_text = response.choices[0].message.content.strip()
            update_community_summary(
                graph, comm_id,
                summary=summary_text,
                community_name=comm.get("semantic_tag", f"Community {comm_id}"),
            )
        except Exception as e:
            logger.warning(f"社区 {comm_id} 摘要生成失败: {e}")

    # 分批并发
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        await asyncio.gather(*[_summarize_one(cid, c) for cid, c in batch])

    logger.info(f"社区摘要生成完成：{len(pending)} 个社区")
    return graph


# ============================================================
# 图谱持久化
# ============================================================

def save_entity_graph(db_id: str, graph: dict[str, Any]) -> str:
    # 保存前剥除 embedding（减少文件体积；读取时重新计算）
    graph_to_save = _strip_embeddings_for_save(graph)
    path = _graph_path(db_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_to_save, f, ensure_ascii=False, indent=2)
    logger.info(
        f"图谱已保存：{path}（{graph['stats'].get('node_count', 0)} 节点，"
        f"{graph['stats'].get('edge_count', 0)} 边）"
    )
    _maybe_sync_to_neo4j(db_id, graph)
    return path


def _strip_embeddings_for_save(graph: dict[str, Any]) -> dict[str, Any]:
    """保存时剥除节点 embedding（大数组），减少磁盘占用"""
    import copy
    g = copy.deepcopy(graph)
    for node in g.get("nodes", {}).values():
        node.pop("embedding", None)
    return g


def load_entity_graph(db_id: str) -> dict[str, Any] | None:
    path = _graph_path(db_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            graph = json.load(f)
        # 读取后重建 embedding（保证内存中有向量用于检索）
        _rebuild_embeddings(graph)
        return graph
    except Exception as e:
        logger.warning(f"加载图谱失败 {db_id}: {e}")
        return None


def _rebuild_embeddings(graph: dict[str, Any]) -> None:
    """加载后重建节点 embedding（若已被 strip）"""
    for node in graph.get("nodes", {}).values():
        if "embedding" not in node:
            node["embedding"] = _compute_node_embedding(
                node.get("type", "Concept"),
                node.get("label", ""),
            )


def graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    if not graph:
        return {"node_count": 0, "edge_count": 0, "paper_count": 0, "node_type_count": {}}
    return graph.get("stats", {})


# ============================================================
# 引用图工具
# ============================================================

def get_citation_chain(
    graph: dict[str, Any],
    paper_id_raw: str,
    direction: str = "out",
    max_hops: int = 3,
) -> list[dict[str, Any]]:
    """
    引用链查询：从目标论文出发追踪引用/被引关系。

    Args:
        graph:        已构建的实体图谱
        paper_id_raw: 起始论文 ID
        direction:    "out" = 该论文引用哪些论文（cites）
                      "in"  = 哪些论文引用了该论文（被引）
        max_hops:     最大追踪跳数

    Returns:
        [{"paper_id": ..., "title": ..., "hop": ..., "path": [...]}, ...]
    """
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])

    cite_edges: list[tuple[str, str]] = [
        (e["source"], e["target"])
        for e in edges if e.get("type") == "cites"
    ]

    # 构建引用索引
    out_index: dict[str, list[str]] = defaultdict(list)
    in_index: dict[str, list[str]] = defaultdict(list)
    for src, tgt in cite_edges:
        out_index[src].append(tgt)
        in_index[tgt].append(src)

    start_node = _entity_id("Paper", paper_id_raw)
    if start_node not in nodes:
        return []

    index = out_index if direction == "out" else in_index
    visited: dict[str, int] = {start_node: 0}
    queue: list[tuple[str, list[str]]] = [(start_node, [start_node])]
    result: list[dict[str, Any]] = []

    while queue:
        current, path = queue.pop(0)
        hop = len(path) - 1
        if hop > 0:
            result.append({
                "paper_id": current,
                "title":    nodes.get(current, {}).get("label", current),
                "hop":      hop,
                "path":     [nodes.get(p, {}).get("label", p) for p in path],
            })
        if hop >= max_hops:
            continue
        for neighbor in index.get(current, []):
            if neighbor not in visited:
                visited[neighbor] = hop + 1
                queue.append((neighbor, path + [neighbor]))

    return sorted(result, key=lambda x: x["hop"])


# ============================================================
# GraphRAG 五分量检索重排（v2）
# ============================================================

def rerank_chunks_by_entity_graph(
    chunks: list[dict[str, Any]],
    graph: dict[str, Any],
    query_text: str,
    top_k: int,
    search_type: str = "local",
) -> list[dict]:
    """
    GraphRAG 五分量检索重排（v2）。

    分量及权重：
        final_score =
            0.35 * vector_score        (向量相似度，来自向量库)
            0.25 * graph_score         (图传播分数)
            0.15 * entity_match_score  (实体精确命中)
            0.15 * embedding_score     (图节点语义向量相似度)
            0.10 * lexical_score       (词法 overlap)

    search_type:
        "local"     — 1 跳，围绕直接连接实体打分
        "community" — 1 跳 + 社区摘要 boost
        "global"    — 3 跳全图传播，适合跨领域总结
    """
    if not chunks or not graph:
        return chunks[:top_k]

    nodes         = graph.get("nodes", {})
    edges         = graph.get("edges", [])
    paper_entities = graph.get("paper_entities", {})
    communities   = graph.get("communities", {})
    query_tokens  = set(_tokenize(query_text))
    query_embed   = _get_query_embedding(query_text)

    # ---- 邻接表 ----
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in edges:
        s, t, w = edge.get("source"), edge.get("target"), float(edge.get("weight", 1.0))
        if s and t:
            adjacency[s].append((t, w))
            adjacency[t].append((s, w))

    # ---- 种子分数 ----
    seed_scores = _extract_query_entity_seeds(query_text, graph)

    # ---- 实体精确命中分数（entity_match_score） ----
    entity_match_scores: dict[str, float] = {}
    for node_id, node in nodes.items():
        norm_label = node.get("norm_label", "")
        label_tokens = set(_tokenize(norm_label))
        if query_tokens and label_tokens:
            precision = len(query_tokens & label_tokens) / len(label_tokens)
            recall    = len(query_tokens & label_tokens) / len(query_tokens)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                entity_match_scores[node_id] = f1 * _type_boost(node.get("type", ""))

    # ---- 图节点嵌入相似度（embedding_score） ----
    embedding_scores: dict[str, float] = {}
    for node_id, node in nodes.items():
        node_embed = node.get("embedding")
        if node_embed:
            embedding_scores[node_id] = _embedding_similarity(query_embed, node_embed)

    # ---- 社区 boost（community / global） ----
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

    # ---- 图传播 ----
    hops  = 3 if search_type == "global" else 1
    alpha = float(config.get("graphrag.seed_alpha") or 0.55)

    scores: dict[str, float] = dict(seed_scores)
    for nid, boost in community_boost.items():
        scores[nid] = scores.get(nid, 0.0) + boost

    for _ in range(max(hops, 1)):
        next_scores = dict(scores)
        for node_id, neighs in adjacency.items():
            if not neighs:
                continue
            total_w   = sum(w for _, w in neighs) or 1.0
            propagated = sum(scores.get(n, 0.0) * w for n, w in neighs) / total_w
            seed       = seed_scores.get(node_id, 0.0)
            next_scores[node_id] = alpha * seed + (1 - alpha) * propagated
        scores = next_scores

    # ---- 给 chunks 打分 ----
    ranked = []
    for chunk in chunks:
        vec_score  = float(chunk.get("score", 0.0) or 0.0)
        metadata   = chunk.get("metadata", {}) or {}
        content    = chunk.get("content", "")
        content_tokens = set(_tokenize(content))
        lexical_score  = (
            len(query_tokens & content_tokens) / max(len(query_tokens), 1)
            if query_tokens else 0.0
        )

        # graph_score（图传播）
        graph_score  = 0.0
        entity_score = 0.0
        emb_score    = 0.0
        score_count  = 0

        paper_id_raw = str(metadata.get("paper_id", "")).strip()
        if paper_id_raw and paper_id_raw in paper_entities:
            ent_ids = paper_entities[paper_id_raw]
            g_vals  = [scores.get(eid, 0.0) for eid in ent_ids]
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
            if mid in scores and scores[mid] > 0:
                graph_score  = max(graph_score,  scores[mid])
                entity_score = max(entity_score, entity_match_scores.get(mid, 0.0))
                emb_score    = max(emb_score,    embedding_scores.get(mid, 0.0))
                score_count += 1

        # 任务/问题直接命中
        core_problem = str(metadata.get("core_problem", "")).strip()
        if core_problem:
            tid = _entity_id("Task", core_problem[:100])
            if tid in scores and scores[tid] > 0:
                graph_score  = max(graph_score,  scores[tid])
                entity_score = max(entity_score, entity_match_scores.get(tid, 0.0))
                emb_score    = max(emb_score,    embedding_scores.get(tid, 0.0))
                score_count += 1

        # 归一化
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


# ============================================================
# 社区上下文提取（用于 LLM 推理）
# ============================================================

def get_community_context(
    graph: dict[str, Any],
    query_text: str,
    top_n: int = 3,
) -> str:
    """
    提取与查询最相关的社区摘要，格式化为 LLM 可读文本块。
    若社区无 LLM 摘要，则使用语义标签 + 关键实体列表作为 fallback。
    """
    communities = graph.get("communities", {})
    if not communities:
        return ""

    query_tokens = set(_tokenize(query_text))
    query_embed  = _get_query_embedding(query_text)
    scored: list[tuple[float, str, str, str]] = []

    for comm_id, comm in communities.items():
        summary    = comm.get("summary", "")
        comm_name  = comm.get("community_name") or comm.get("semantic_tag") or f"Community {comm_id}"
        sem_tag    = comm.get("semantic_tag", "")

        if not summary:
            summary = (
                f"[{sem_tag}] "
                "Key entities: " + ", ".join(comm.get("key_entities", [])[:10])
            )

        comm_tokens = set(_tokenize(summary + " " + comm_name))
        text_overlap = (
            len(query_tokens & comm_tokens) / max(len(query_tokens), 1)
            if query_tokens else 0.0
        )
        # 额外：社区代表节点 embedding 相似度
        embed_sim = 0.0
        node_ids  = comm.get("nodes", [])
        nodes     = graph.get("nodes", {})
        if node_ids:
            sample_embeds = [
                nodes[nid]["embedding"]
                for nid in node_ids[:8]
                if nid in nodes and "embedding" in nodes[nid]
            ]
            if sample_embeds:
                avg_sim = sum(_embedding_similarity(query_embed, e) for e in sample_embeds)
                embed_sim = avg_sim / len(sample_embeds)

        combined_score = 0.6 * text_overlap + 0.4 * embed_sim
        scored.append((combined_score, comm_name, summary, sem_tag))

    scored.sort(reverse=True)
    parts = []
    for _, name, summary, tag in scored[:top_n]:
        if summary:
            parts.append(f"[Research Community: {name}]\n{summary}")

    return "\n\n".join(parts)


# ============================================================
# 局部子图语义化（结构化知识块）
# ============================================================

def get_local_subgraph_context(
    graph: dict[str, Any],
    query_text: str,
    max_hops: int = 2,
    max_triples: int = 40,
) -> str:
    """
    提取与查询最相关实体的局部子图，转换为结构化可读知识块。

    输出示例：
        [Entity: Transformer | Type: Method]
          → proposes       : Attention Mechanism
          → evaluated on   : WMT14
          → achieves       : BLEU 28.4
          → solves         : Machine Translation

    这种格式比原始 "A --[rel]--> B" 更适合 LLM 直接推理。
    """
    seed_scores = _extract_query_entity_seeds(query_text, graph)
    if not seed_scores:
        return ""

    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])

    # 取 top-5 种子
    top_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    seed_ids  = {nid for nid, _ in top_seeds}

    # BFS 扩展
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

    # 收集子图边
    subgraph_edges = [
        e for e in edges
        if e["source"] in visited and e["target"] in visited
    ][:max_triples]

    if not subgraph_edges:
        return ""

    # 构建以"源节点"为核心的出边字典
    out_edges: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for e in subgraph_edges:
        src_id    = e["source"]
        dst_label = nodes.get(e["target"], {}).get("label", e["target"])
        rel_label = RELATION_LABELS.get(e["type"], e["type"])
        dst_type  = nodes.get(e["target"], {}).get("type", "")
        out_edges[src_id].append((rel_label, dst_label, dst_type))

    lines = ["[Local Knowledge Subgraph]"]
    rendered_nodes: set[str] = set()

    # 优先输出种子节点
    ordered_nodes = list(seed_ids) + [n for n in visited if n not in seed_ids]

    for node_id in ordered_nodes:
        if node_id not in out_edges:
            continue
        if node_id in rendered_nodes:
            continue
        rendered_nodes.add(node_id)

        node_info = nodes.get(node_id, {})
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


def get_multi_hop_reasoning_path(
    graph: dict[str, Any],
    query_text: str,
    max_hops: int = 3,
) -> str:
    """
    多跳推理路径提取：找到从查询实体出发的最具信息量的路径链。

    返回格式：
        [Multi-hop Reasoning Paths]
        Path 1: Transformer → solves → Machine Translation → measured_by → BLEU
        Path 2: BERT → improves upon → Transformer → evaluated on → GLUE
    """
    seed_scores = _extract_query_entity_seeds(query_text, graph)
    if not seed_scores:
        return ""

    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])

    # 构建有向边字典
    out_edges: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for e in edges:
        src, tgt, rel = e.get("source"), e.get("target"), e.get("type", "related_to")
        if src and tgt:
            out_edges[src].append((rel, tgt, str(e.get("weight", 0.5))))

    top_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    all_paths: list[tuple[float, list[str]]] = []

    for start_node, seed_score in top_seeds:
        # DFS 寻路（剪枝：路径分数下降太快则停止）
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
                edge_w = float(w_str) if w_str else 0.5
                new_score = path_score * edge_w * _type_boost(nodes.get(neighbor, {}).get("type", ""))
                if new_score < 0.05:
                    continue
                path_repr = path + [f"--[{rel}]-->", neighbor]
                stack.append((neighbor, path_repr, new_score))

    # 按分数排序，去重，取 top-5
    all_paths.sort(key=lambda x: x[0], reverse=True)
    seen_paths: set[str] = set()
    lines = ["[Multi-hop Reasoning Paths]"]
    path_count = 0
    for score, path in all_paths:
        if path_count >= 5:
            break
        # 转换为可读字符串
        readable_parts = []
        for segment in path:
            if segment.startswith("--["):
                rel_name = segment[3:-3]
                readable_parts.append(f" {RELATION_LABELS.get(rel_name, rel_name)} ")
            else:
                label = nodes.get(segment, {}).get("label", segment)
                readable_parts.append(label)
        readable = "→".join(readable_parts)
        if readable not in seen_paths:
            seen_paths.add(readable)
            lines.append(f"  Path {path_count + 1}: {readable}")
            path_count += 1

    return "\n".join(lines) if path_count > 0 else ""


# ============================================================
# Neo4j 同步（可选）
# ============================================================

def _maybe_sync_to_neo4j(db_id: str, graph: dict[str, Any]) -> None:
    """同步图谱到 Neo4j（含新增节点类型和关系类型）"""
    try:
        neo4j_conf = config.get("neo4j") or {}
    except Exception:
        neo4j_conf = {}

    if not neo4j_conf or not neo4j_conf.get("enable"):
        return
    if GraphDatabase is None:
        logger.warning("neo4j.sync 已启用但未安装驱动，请 `pip install neo4j`")
        return

    uri      = str(neo4j_conf.get("uri") or "neo4j://localhost:7687")
    user     = str(neo4j_conf.get("user") or "neo4j")
    password = str(neo4j_conf.get("password") or "123456")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception as e:
        logger.warning(f"连接 Neo4j 失败（{uri}）: {e}")
        return

    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])

    def _sync(tx: Any) -> None:
        tx.run("MATCH (n:Entity {db_id: $db_id}) DETACH DELETE n", db_id=db_id)
        for node in nodes.values():
            tx.run(
                """
                MERGE (n:Entity {id: $id, db_id: $db_id})
                SET n.label      = $label,
                    n.type       = $type,
                    n.norm_label = $norm_label,
                    n.description = $description
                """,
                id=node.get("id"),
                label=node.get("label"),
                type=node.get("type"),
                norm_label=node.get("norm_label", ""),
                description=node.get("description", ""),
                db_id=db_id,
            )
        for edge in edges:
            tx.run(
                """
                MATCH (s:Entity {id: $src, db_id: $db_id})
                MATCH (t:Entity {id: $dst, db_id: $db_id})
                MERGE (s)-[r:RELATION {type: $type}]->(t)
                SET r.weight = $weight
                """,
                src=edge.get("source"),
                dst=edge.get("target"),
                type=edge.get("type"),
                weight=float(edge.get("weight", 1.0) or 1.0),
                db_id=db_id,
            )

    try:
        with driver.session() as session:
            session.execute_write(_sync)
        logger.info(f"图谱已同步到 Neo4j（db_id={db_id}, 节点={len(nodes)}, 边={len(edges)}）")
    except Exception as e:
        logger.warning(f"同步图谱到 Neo4j 失败: {e}")
    finally:
        driver.close()


# ============================================================
# 内部工具
# ============================================================

def _update_stats(graph: dict[str, Any]) -> None:
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


def _extract_query_entity_seeds(
    query_text: str,
    graph: dict[str, Any],
) -> dict[str, float]:
    """
    三阶段种子提取：
    1. 别名精确匹配
    2. 缩写展开后匹配
    3. Token overlap 补充
    """
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
                seeds[exp_node_id] = max(seeds.get(exp_node_id, 0.0), 0.9 * _type_boost(node_type))

    # Stage 3: Token overlap 补充
    for node_id, node in nodes.items():
        label_tokens = set(_tokenize(node.get("norm_label", "")))
        if not label_tokens:
            continue
        overlap = len(query_tokens & label_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
        if overlap > 0:
            seeds[node_id] = max(
                seeds.get(node_id, 0.0),
                overlap * _type_boost(node.get("type", "")),
            )

    return seeds
