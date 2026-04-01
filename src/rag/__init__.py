"""
RAG 子系统：向量知识库、文档解析入库。

- ``src.rag``：``KnowledgeBaseManager``、全局 ``knowledge_base``、路由等
- ``src.rag.retrieval``：再导出 ``src.retriever``（``retrieval_tool`` 等）
"""

from __future__ import annotations

import os
from pathlib import Path

from src.core.config import config

__all__ = ["config", "knowledge_base"]


def _kb_work_root() -> str:
    save_dir = config.get("SAVE_DIR")
    if not save_dir:
        save_dir = str(Path(__file__).resolve().parents[2] / "data")
    return os.path.join(str(save_dir), "knowledge_base_data")


from src.rag.factory import KnowledgeBaseFactory
from src.rag.implementations.chroma import ChromaKB
from src.rag.manager import KnowledgeBaseManager

KnowledgeBaseFactory.register("chroma", ChromaKB, default_config={})

knowledge_base = KnowledgeBaseManager(_kb_work_root())
