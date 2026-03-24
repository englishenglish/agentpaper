from __future__ import annotations

import sys
import os
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autogen_agentchat.agents import AssistantAgent
from src.utils.log_utils import setup_logger
from src.core.model_client import create_qa_model_client
from src.core.state_models import State, ExecutionState, BackToFrontData
from src.core.prompts import qa_agent_prompt, graphrag_query_prompt, hybrid_query_prompt
from src.services.retrieval_tool import retrieval_tool
from src.services.graph_store import (
    load_entity_graph,
    get_community_context,
    get_local_subgraph_context,
    get_multi_hop_reasoning_path,
)
from src.core.config import config
from src.knowledge.knowledge import knowledge_base as kb_manager

logger = setup_logger(__name__)

# ============================================================
# 1. 初始化问答 Agent
# ============================================================
qa_model_client = create_qa_model_client()


def _make_qa_agent() -> AssistantAgent:
    """每次调用创建新实例，避免多并发请求共享对话历史导致上下文污染"""
    return AssistantAgent(
        name="qa_agent",
        model_client=qa_model_client,
        system_message=qa_agent_prompt,
        model_client_stream=True,
    )


def _make_graphrag_agent() -> AssistantAgent:
    """每次调用创建新实例，避免多并发请求共享对话历史导致上下文污染"""
    return AssistantAgent(
        name="graphrag_agent",
        model_client=qa_model_client,
        system_message=graphrag_query_prompt,
        model_client_stream=True,
    )


def _make_hybrid_agent() -> AssistantAgent:
    """RAG + GraphRAG 混合模式 Agent（both 模式专用）"""
    return AssistantAgent(
        name="hybrid_agent",
        model_client=qa_model_client,
        system_message=hybrid_query_prompt,
        model_client_stream=True,
    )


# ============================================================
# 2. 核心检索逻辑
# ============================================================

async def retrieve_context(
    query: str,
    top_k: int = 8,
    preferred_db_ids: List[str] | None = None,
    retrieval_mode: str = "rag",
) -> Tuple[str, List[str]]:
    """
    统一检索入口：支持 rag / graphrag / both 三种模式。

    返回 (context_str, formatted_passages)。
    每个 passage 已经包含 [Paper ID | Title | Section] 元数据头部，
    LLM 可以直接从中提取 paper_id / title 用于引用。
    """
    try:
        passages = await retrieval_tool(
            [query],
            preferred_db_ids=preferred_db_ids,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        if not passages:
            logger.warning("知识库中未检索到相关内容。")
            return "未检索到与该问题相关的文献片段。", []

        selected = passages[:top_k]
        # 段落之间加空行，便于 LLM 区分边界
        context_str = "\n\n".join(selected)
        return context_str, selected
    except Exception as e:
        logger.error(f"检索文献时发生错误: {e}")
        return "", []


def _build_graphrag_context(
    query: str,
    db_ids: List[str],
    search_type: str,
) -> str:
    """
    构建 GraphRAG 三级检索上下文文本。

    search_type:
        "local"     — 局部子图 + 直接邻居
        "community" — 社区摘要匹配
        "global"    — 全图多跳传播
    """
    parts: list[str] = []

    for db_id in db_ids:
        graph = load_entity_graph(db_id)
        if not graph:
            continue

        if search_type == "local":
            subgraph_text = get_local_subgraph_context(graph, query, max_hops=1)
            if subgraph_text:
                parts.append(subgraph_text)

        elif search_type == "community":
            community_text = get_community_context(graph, query, top_n=3)
            if community_text:
                parts.append(community_text)
            # 也附上局部子图
            subgraph_text = get_local_subgraph_context(graph, query, max_hops=2)
            if subgraph_text:
                parts.append(subgraph_text)

        else:  # global
            community_text = get_community_context(graph, query, top_n=5)
            if community_text:
                parts.append(community_text)
            subgraph_text = get_local_subgraph_context(graph, query, max_hops=3)
            if subgraph_text:
                parts.append(subgraph_text)
            multihop_text = get_multi_hop_reasoning_path(graph, query, max_hops=3)
            if multihop_text:
                parts.append(multihop_text)

    return "\n\n".join(parts)


def _resolve_search_type(retrieval_mode: str) -> str:
    """将前端 retrieval_mode 映射到 GraphRAG search_type"""
    mapping = {
        "graphrag_local": "local",
        "graphrag_community": "community",
        "graphrag_global": "global",
        "graphrag": "local",   # 默认 local
        "both": "community",   # both 模式使用 community
        "rag": "local",
    }
    return mapping.get(retrieval_mode, "local")


# ============================================================
# 3. LangGraph 节点入口
# ============================================================

async def qa_node(state: State) -> State:
    """
    问答系统节点：
        rag 模式    → 纯向量检索 + qa_agent
        graphrag 模式 → 三级图谱检索 + graphrag_agent
        both 模式   → 向量检索 + 社区图谱 + graphrag_agent
    """
    state_queue = state["state_queue"]
    current_state = state["value"]
    cfg = getattr(current_state, "config", None) or {}
    if cfg.get("bypass_to_qa"):
        current_state.config = {**cfg, "bypass_to_qa": False}

    current_state.current_step = ExecutionState.QA_ANSWERING

    await state_queue.put(
        BackToFrontData(step=ExecutionState.QA_ANSWERING, state="initializing", data="准备分析您的问题...")
    )

    # ---- 获取用户问题 ----
    user_question = getattr(current_state, "current_question", None) or current_state.user_request

    # ---- 解析配置 ----
    cfg = getattr(current_state, "config", {}) or {}
    preferred_db_ids: List[str] = []
    if isinstance(cfg.get("selected_db_ids"), list):
        preferred_db_ids = cfg["selected_db_ids"]
    elif isinstance(cfg.get("auto_selected_db_ids"), list):
        preferred_db_ids = cfg["auto_selected_db_ids"]

    retrieval_mode: str = cfg.get("retrieval_mode", "rag")
    use_graphrag = retrieval_mode in ("graphrag", "graphrag_local", "graphrag_community", "graphrag_global", "both")

    # ---- 通知前端正在使用哪些知识库 ----
    if preferred_db_ids:
        try:
            all_dbs = kb_manager.get_databases().get("databases", [])
            db_name_map = {db["db_id"]: db.get("name", db["db_id"]) for db in all_dbs}
        except Exception:
            db_name_map = {}
        used_kbs = [
            {"db_id": db_id, "name": db_name_map.get(db_id, db_id)}
            for db_id in preferred_db_ids
        ]
        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.QA_ANSWERING,
                state="kb_context",
                data=used_kbs,
            )
        )

    # ---- 向量检索 ----
    await state_queue.put(
        BackToFrontData(step=ExecutionState.QA_ANSWERING, state="retrieving", data="正在检索相关文献片段...")
    )
    context_str, raw_contexts = await retrieve_context(
        user_question,
        top_k=5,
        preferred_db_ids=preferred_db_ids,
        retrieval_mode=retrieval_mode,
    )
    current_state.retrieved_contexts = raw_contexts

    if not context_str:
        context_str = "未检索到与该问题相关的文献片段。"

    # ---- 组装对话历史 ----
    history_prompt = ""
    if current_state.chat_history:
        history_prompt = "【历史对话记录】\n"
        for msg in current_state.chat_history[-4:]:
            role_name = "用户" if msg.get("role") == "user" else "AI助手"
            history_prompt += f"{role_name}：{msg.get('content')}\n"
        history_prompt += "\n====================\n\n"

    # ---- 构建 Task Prompt ----
    if use_graphrag:
        search_type = _resolve_search_type(retrieval_mode)
        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.QA_ANSWERING,
                state="graph_retrieving",
                data=f"正在执行 GraphRAG {search_type} 检索...",
            )
        )

        graph_context = _build_graphrag_context(user_question, preferred_db_ids, search_type)

        # both 模式：RAG 文献片段 + 图谱上下文同时提供，使用专用 hybrid_agent
        # graphrag_* 模式：以图谱上下文为主，文献片段作为补充
        if retrieval_mode == "both":
            task_prompt = (
                f"【Retrieved Paper Passages (RAG)】\n{context_str}\n\n"
                f"════════════════════════════════════\n\n"
                f"【Knowledge Graph Context (GraphRAG)】\n{graph_context}\n\n"
                f"════════════════════════════════════\n\n"
                f"{history_prompt}"
                f"【User Question】\n{user_question}"
            )
            active_agent = _make_hybrid_agent()
        else:
            task_prompt = (
                f"【Retrieved Paper Passages】\n{context_str}\n\n"
                f"════════════════════════════════════\n\n"
                f"【Knowledge Graph Context】\n{graph_context}\n\n"
                f"════════════════════════════════════\n\n"
                f"{history_prompt}"
                f"【User Question】\n{user_question}"
            )
            active_agent = _make_graphrag_agent()
    else:
        task_prompt = (
            f"【检索到的文献片段】\n{context_str}\n\n"
            f"====================\n\n"
            f"{history_prompt}"
            f"【当前用户问题】\n{user_question}"
        )
        active_agent = _make_qa_agent()

    # ---- 调用 LLM 生成回答 ----
    await state_queue.put(
        BackToFrontData(step=ExecutionState.QA_ANSWERING, state="generating", data="正在生成回答...")
    )
    try:
        result = await active_agent.run(task=task_prompt)
        final_answer = result.messages[-1].content
    except Exception as e:
        logger.error(f"问答生成失败: {str(e)}")
        final_answer = f"抱歉，生成回答时出现错误：{str(e)}"
        current_state.error.qa_node_error = str(e)

    current_state.qa_answer = final_answer
    await state_queue.put(
        BackToFrontData(step=ExecutionState.QA_ANSWERING, state="completed", data=final_answer)
    )
    return {"value": current_state}
