"""
相邻句向量相似度切块：用 Embedding 模型编码句子，按相邻余弦相似度决定是否合并为同一块。

与「按 Markdown 标题切块」不同：此处依赖向量空间中的语义连续性。
"""

from __future__ import annotations

import re
from typing import Any
import pysbd
from src.core.config import config
from src.core.embedding import embedding_cosine_similarity, get_shared_embedder
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)
_segmenter = pysbd.Segmenter(language="en", clean=False)

def split_into_sentences(text: str) -> list[str]:
    """使用 pysbd 进行高精度语义分句，专门应对学术文献中的缩写、小数点和引用。"""
    if not text or not text.strip():
        return []
        
    t = text.replace("\r\n", "\n").strip()
    
    try:
        # pysbd 直接进行边界划分
        sentences = _segmenter.segment(t)
    except Exception as e:
        logger.warning(f"pysbd 分句失败，回退到按换行符粗切: {e}")
        sentences = t.split("\n")
    
    out: list[str] = []
    for p in sentences:
        p = p.strip()
        if not p:
            continue
            
        # 保留您原有的长句兜底逻辑：如果解析出的单句依然极其长且包含换行，则按换行硬切
        if len(p) > 800 and "\n" in p:
            for line in p.split("\n"):
                line = line.strip()
                if line and len(line) >= 2:
                    out.append(line)
        else:
            out.append(p)
            
    return out if out else [t]

def _all_near_zero(vec: list[float]) -> bool:
    return not vec or sum(abs(x) for x in vec) < 1e-8


def embedding_sentence_chunk_chunks(
    text: str,
    file_id: str,
    filename: str,
    params: dict | None = None,
) -> list[dict[str, Any]]:
    """
    返回与 Chroma 入库所需的 chunk 列表（content / id / metadata 等）。

    参数（params 或 config）：
        embedding_chunk_adjacent_threshold: 相邻句余弦相似度低于此则新开一块（0~1）
        embedding_chunk_max_chars: 单块最大字符（硬上限）
        embedding_chunk_embed_batch_size: 批量编码条数
    """
    params = params or {}
    threshold = float(
        params.get("embedding_chunk_adjacent_threshold")
        or config.get("embedding_chunk_adjacent_threshold", 0.42)
    )
    max_chars = int(
        params.get("embedding_chunk_max_chars") or config.get("embedding_chunk_max_chars", 4000)
    )
    batch_size = int(
        params.get("embedding_chunk_embed_batch_size")
        or config.get("embedding_chunk_embed_batch_size", 64)
    )
    threshold = max(0.0, min(1.0, threshold))

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        body = sentences[0].strip()
        meta = _meta(file_id, filename, 0)
        return [
            {
                "content": body,
                "id": f"{file_id}_chunk_0",
                "source": filename,
                "chunk_id": f"{file_id}_chunk_0",
                "chunk_type": "embedding_semantic",
                "metadata": meta,
            }
        ]

    embedder = get_shared_embedder()
    embeddings: list[list[float]] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        embeddings.extend(embedder.get_embeddings(batch))

    if len(embeddings) != len(sentences):
        logger.warning("embedding_sentence_chunk: 向量条数与句子数不一致，回退为整段一块")
        return _single_fallback_chunk(text, file_id, filename)

    if all(_all_near_zero(e) for e in embeddings):
        logger.warning("embedding_sentence_chunk: 嵌入全零，回退为整段一块")
        return _single_fallback_chunk(text, file_id, filename)

    groups: list[list[str]] = []
    current: list[str] = [sentences[0]]

    for i in range(len(sentences) - 1):
        sim = embedding_cosine_similarity(embeddings[i], embeddings[i + 1])
        next_s = sentences[i + 1]
        candidate = "\n".join(current + [next_s])

        if len(candidate) > max_chars:
            groups.append(current)
            current = [next_s]
        elif sim < threshold:
            groups.append(current)
            current = [next_s]
        else:
            current.append(next_s)

    groups.append(current)

    chunks: list[dict[str, Any]] = []
    for idx, g in enumerate(groups):
        body = "\n".join(g).strip()
        if not body:
            continue
        meta = _meta(file_id, filename, idx)
        chunks.append(
            {
                "content": body,
                "id": f"{file_id}_chunk_{idx}",
                "source": filename,
                "chunk_id": f"{file_id}_chunk_{idx}",
                "chunk_type": "embedding_semantic",
                "metadata": meta,
            }
        )

    logger.info(f"Embedding adjacent-sentence chunking: {len(chunks)} chunks for {filename}")
    return chunks


def _meta(file_id: str, filename: str, idx: int) -> dict[str, Any]:
    return {
        "full_doc_id": file_id,
        "chunk_id": f"{file_id}_chunk_{idx}",
        "chunk_index": idx,
        "source": filename,
    }


def _single_fallback_chunk(text: str, file_id: str, filename: str) -> list[dict[str, Any]]:
    body = text.strip()
    meta = _meta(file_id, filename, 0)
    return [
        {
            "content": body,
            "id": f"{file_id}_chunk_0",
            "source": filename,
            "chunk_id": f"{file_id}_chunk_0",
            "chunk_type": "embedding_semantic",
            "metadata": meta,
        }
    ]
