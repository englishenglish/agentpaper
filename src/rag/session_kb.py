"""会话级联网建库：自动创建与本对话绑定的知识库（单库）。"""

from __future__ import annotations

from src.rag import knowledge_base
from src.core.config import config


async def create_session_research_kb(user_request: str) -> str:
    """未选手动知识库时，为联网检索流程创建唯一会话知识库。"""
    embedding_cfg = config.get("embedding-model") or {}
    embedding_provider = embedding_cfg.get("model-provider")
    provider_dic = config.get(embedding_provider) or {}
    embed_info = {
        "name": embedding_cfg.get("model"),
        "dimension": embedding_cfg.get("dimension"),
        "base_url": provider_dic.get("base_url"),
        "api_key": provider_dic.get("api_key"),
    }
    kb_type = config.get("KB_TYPE") or "chroma"
    raw = (user_request or "对话").strip().replace("\n", " ")
    safe = raw[:36] + ("…" if len(raw) > 36 else "")
    name = f"会话检索-{safe}"
    db_info = await knowledge_base.create_database(
        name,
        "由联网检索自动创建，与本会话绑定",
        kb_type=kb_type,
        embed_info=embed_info,
        llm_info=None,
    )
    return db_info["db_id"]
