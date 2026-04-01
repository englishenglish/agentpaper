"""
graph_schema.py — 知识图谱共享常量与工具函数

graph_builder.py 和 graphrag_retriever.py 共用的实体/关系类型、
缩写映射、文本归一化工具统一在此定义，避免重复和漂移。
"""
from __future__ import annotations

import re

# ============================================================
# Schema 常量
# ============================================================

VALID_ENTITY_TYPES: set[str] = {
    "Paper", "Method", "Model", "Task", "Dataset",
    "Metric", "Experiment", "Result", "Contribution", "Concept",
}

VALID_RELATION_TYPES: set[str] = {
    "proposes", "improves", "uses", "evaluates_on",
    "compared_with", "achieves", "applied_to", "related_to",
    "has_experiment", "uses_dataset", "measures", "produces",
    "has_contribution", "solves", "cites", "extends",
}

RELATION_WEIGHT: dict[str, float] = {
    "proposes":         1.0,
    "has_contribution": 1.0,
    "solves":           0.95,
    "improves":         0.95,
    "evaluates_on":     0.9,
    "produces":         0.9,
    "measures":         0.88,
    "has_experiment":   0.85,
    "achieves":         0.85,
    "uses_dataset":     0.82,
    "uses":             0.8,
    "applied_to":       0.8,
    "cites":            0.75,
    "extends":          0.75,
    "compared_with":    0.7,
    "related_to":       0.5,
}

RELATION_LABELS: dict[str, str] = {
    "proposes":         "proposes",
    "improves":         "improves upon",
    "uses":             "uses",
    "evaluates_on":     "evaluated on",
    "compared_with":    "compared with",
    "achieves":         "achieves",
    "applied_to":       "applied to",
    "related_to":       "related to",
    "has_experiment":   "has experiment",
    "uses_dataset":     "uses dataset",
    "measures":         "measures",
    "produces":         "produces result",
    "has_contribution": "contributes",
    "solves":           "solves",
    "cites":            "cites",
    "extends":          "extends",
}

TYPE_BOOST: dict[str, float] = {
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

ABBREVIATION_MAP: dict[str, str] = {
    "bert":         "bidirectional encoder representations from transformers",
    "gpt":          "generative pre-trained transformer",
    "gpt-2":        "generative pre-trained transformer 2",
    "gpt-3":        "generative pre-trained transformer 3",
    "gpt-4":        "generative pre-trained transformer 4",
    "gpt4":         "generative pre-trained transformer 4",
    "t5":           "text-to-text transfer transformer",
    "llm":          "large language model",
    "llms":         "large language models",
    "nlp":          "natural language processing",
    "cv":           "computer vision",
    "rl":           "reinforcement learning",
    "dl":           "deep learning",
    "ml":           "machine learning",
    "gan":          "generative adversarial network",
    "cnn":          "convolutional neural network",
    "rnn":          "recurrent neural network",
    "lstm":         "long short-term memory",
    "transformer":  "transformer",
    "attention":    "attention mechanism",
    "rag":          "retrieval augmented generation",
    "graphrag":     "graph retrieval augmented generation",
    "kg":           "knowledge graph",
    "kgs":          "knowledge graphs",
    "qa":           "question answering",
    "mt":           "machine translation",
    "nmt":          "neural machine translation",
    "bleu":         "bilingual evaluation understudy",
    "rouge":        "recall oriented understudy for gisting evaluation",
    "f1":           "f1 score",
    "acc":          "accuracy",
    "sota":         "state of the art",
    "finetune":     "fine-tuning",
    "fine-tune":    "fine-tuning",
    "pretraining":  "pre-training",
    "pre-training": "pre-training",
    "zero-shot":    "zero-shot learning",
    "few-shot":     "few-shot learning",
}

MODEL_VARIANT_PATTERN: re.Pattern = re.compile(
    r"[-_](base|large|small|medium|xl|xxl|uncased|cased|"
    r"v\d+[\.\d]*|\d+[bm]|instruct|chat|hf|en|zh|multilingual)$",
    re.IGNORECASE,
)


# ============================================================
# 共享工具函数
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
    return RELATION_WEIGHT.get(rel, 0.5)


def type_boost(node_type: str) -> float:
    return TYPE_BOOST.get(node_type, 1.0)


def expand_abbreviation(text: str) -> str:
    return ABBREVIATION_MAP.get(text.lower(), text)


def strip_model_variant(name: str) -> str:
    stripped = MODEL_VARIANT_PATTERN.sub("", name.strip())
    return stripped if stripped else name


def jaccard_sim(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
