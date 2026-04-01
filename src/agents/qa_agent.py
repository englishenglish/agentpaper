from __future__ import annotations

from typing import Any, List, Tuple

from autogen_agentchat.agents import AssistantAgent
from src.utils.log_utils import setup_logger
from src.core.model_client import create_qa_model_client
from src.core.state_models import State, ExecutionState, BackToFrontData
from src.core.prompts import qa_agent_prompt, graphrag_query_prompt, hybrid_query_prompt
from src.rag.retrieval import retrieval_tool, normalize_retrieval_mode
from src.core.embedding import get_shared_embedder
from src.graphrag.graph_builder import load_entity_graph
from src.retriever import GraphRAGRetriever
from src.core.config import config
from src.rag import knowledge_base as kb_manager
from src.agents.streaming_utils import stream_assistant_to_queue

logger = setup_logger(__name__)

_get_embedder = get_shared_embedder


def _truncate_text(text: str, max_chars: int, label: str = "") -> str:
    """按字符截断，避免单次请求超过模型上下文。"""
    if max_chars <= 0 or not text:
        return text or ""
    if len(text) <= max_chars:
        return text
    cut = max_chars - 80
    if cut < 100:
        cut = max_chars
    out = text[:cut].rstrip() + "\n\n…（内容过长已截断"
    if label:
        out += f"：{label}"
    out += "）…"
    return out


def _apply_qa_prompt_limits(
    context_str: str,
    graph_context: str,
    history_prompt: str,
) -> tuple[str, str, str]:
    """按配置限制 RAG / 图谱 / 历史 长度，并在仍超总上限时继续压缩。"""
    max_rag = config.get_int("qa.max_rag_context_chars", 15000)
    max_graph = config.get_int("qa.max_graph_context_chars", 6000)
    max_hist = config.get_int("qa.max_history_chars", 2000)
    max_total = config.get_int("qa.max_combined_context_chars", 22000)

    context_str = _truncate_text(context_str or "", max_rag, "检索片段")
    graph_context = _truncate_text(graph_context or "", max_graph, "图谱上下文")
    history_prompt = _truncate_text(history_prompt or "", max_hist, "历史对话")

    total = len(context_str) + len(graph_context) + len(history_prompt)
    if total <= max_total:
        return context_str, graph_context, history_prompt

    # 仍超长：优先压缩图谱，再历史，最后 RAG
    over = total - max_total
    logger.warning(
        "[QA] 上下文仍超长（约 %s 字，超出 %s），将依次截断图谱/历史/RAG",
        total,
        over,
    )
    if len(graph_context) > 400:
        graph_context = _truncate_text(
            graph_context, max(400, len(graph_context) - over // 3), "图谱上下文"
        )
    total = len(context_str) + len(graph_context) + len(history_prompt)
    if total > max_total:
        over = total - max_total
        if len(history_prompt) > 200:
            history_prompt = _truncate_text(
                history_prompt, max(200, len(history_prompt) - over // 3), "历史"
            )
    total = len(context_str) + len(graph_context) + len(history_prompt)
    if total > max_total:
        over = total - max_total
        budget = max(500, len(context_str) - over)
        context_str = _truncate_text(context_str, budget, "检索片段")
    return context_str, graph_context, history_prompt


# ============================================================
# 1. 初始化问答 Agent
# ============================================================
_qa_model_client = None


def _get_qa_model_client():
    global _qa_model_client
    if _qa_model_client is None:
        _qa_model_client = create_qa_model_client()
    return _qa_model_client


def _make_qa_agent() -> AssistantAgent:
    return AssistantAgent(
        name="qa_agent",
        model_client=_get_qa_model_client(),
        system_message=qa_agent_prompt,
        model_client_stream=True,
    )


def _make_graphrag_agent() -> AssistantAgent:
    return AssistantAgent(
        name="graphrag_agent",
        model_client=_get_qa_model_client(),
        system_message=graphrag_query_prompt,
        model_client_stream=True,
    )


def _make_hybrid_agent() -> AssistantAgent:
    return AssistantAgent(
        name="hybrid_agent",
        model_client=_get_qa_model_client(),
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
) -> Tuple[str, List[str], List[dict[str, Any]]]:
    """
    统一检索入口：支持 rag / graphrag / both 三种模式。

    返回 (context_str, formatted_passages, citation_chunks)。
    context_str 中每个片段带有「### 检索片段 [n]」编号，供模型用 [n] 引用；
    citation_chunks 与编号一一对应，供前端悬停展示 chunk 详情。
    """
    try:
        logger.info(
            "[QA] 调用检索 | top_k=%s | mode=%s | db_ids=%s",
            top_k,
            normalize_retrieval_mode(retrieval_mode),
            preferred_db_ids if preferred_db_ids is not None else "(走 config 默认)",
        )
        passages, citations = await retrieval_tool(
            [query],
            preferred_db_ids=preferred_db_ids,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        if not passages:
            logger.warning("知识库中未检索到相关内容。")
            return "未检索到与该问题相关的文献片段。", [], []

        selected = passages[:top_k]
        selected_citations = citations[:top_k]
        for i, c in enumerate(selected_citations, start=1):
            c["ref"] = i
        logger.info(
            "[QA] 检索返回 %s 条片段，送入 LLM 使用前 %s 条",
            len(passages),
            len(selected),
        )
        max_chunk = config.get_int("qa.max_chunk_chars", 5000)
        numbered_blocks = []
        for i, (p, c) in enumerate(zip(selected, selected_citations), start=1):
            cid = c.get("chunk_id", "")
            body = _truncate_text(p, max_chunk, f"片段{i}")
            numbered_blocks.append(
                f"### 检索片段 [{i}]\n**Chunk ID:** `{cid}`\n{body}"
            )
        context_str = "\n\n".join(numbered_blocks)
        return context_str, selected, selected_citations
    except Exception as e:
        logger.error(f"检索文献时发生错误: {e}")
        return "", [], []


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

    embedder = _get_embedder()
    for db_id in db_ids:
        graph = load_entity_graph(db_id, embedder=embedder)
        if not graph:
            continue

        retriever = GraphRAGRetriever(graph, embedder)

        if search_type == "local":
            subgraph_text = retriever.get_local_subgraph_context(query, max_hops=1)
            if subgraph_text:
                parts.append(subgraph_text)

        elif search_type == "community":
            community_text = retriever.get_community_context(query, top_n=3)
            if community_text:
                parts.append(community_text)
            subgraph_text = retriever.get_local_subgraph_context(query, max_hops=2)
            if subgraph_text:
                parts.append(subgraph_text)

        else:  # global
            community_text = retriever.get_community_context(query, top_n=5)
            if community_text:
                parts.append(community_text)
            subgraph_text = retriever.get_local_subgraph_context(query, max_hops=3)
            if subgraph_text:
                parts.append(subgraph_text)
            multihop_text = retriever.get_multi_hop_paths(query, max_hops=3)
            if multihop_text:
                parts.append(multihop_text)

    return "\n\n".join(parts)


def _resolve_search_type(retrieval_mode: str) -> str:
    """将 retrieval_mode（rag / graphrag / both）映射到 GraphRAG search_type。"""
    m = normalize_retrieval_mode(retrieval_mode)
    if m == "both":
        return "community"
    if m == "graphrag":
        return "local"
    return "local"


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
    cfg = getattr(current_state, "config", None) or {}

    current_state.current_step = ExecutionState.QA_ANSWERING

    await state_queue.put(
        BackToFrontData(step=ExecutionState.QA_ANSWERING, state="initializing", data="准备分析您的问题...")
    )

    await state_queue.put(
        BackToFrontData(
            step=ExecutionState.QA_ANSWERING,
            state="session_profile",
            data={
                "kb_binding": cfg.get("kb_binding"),
                "selected_db_ids": list(cfg.get("selected_db_ids") or []),
                "enable_web_search": bool(cfg.get("enable_web_search")),
            },
        )
    )

    # ---- 获取用户问题 ----
    user_question = getattr(current_state, "current_question", None) or current_state.user_request

    # ---- 解析配置 ----
    cfg = getattr(current_state, "config", {}) or {}
    preferred_db_ids: List[str] = []
    if isinstance(cfg.get("selected_db_ids"), list) and cfg["selected_db_ids"]:
        preferred_db_ids = [cfg["selected_db_ids"][0]]

    retrieval_mode = normalize_retrieval_mode(cfg.get("retrieval_mode", "rag"))
    use_graphrag = retrieval_mode in ("graphrag", "both")

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
    context_str, raw_contexts, citation_chunks = await retrieve_context(
        user_question,
        top_k=20,
        preferred_db_ids=preferred_db_ids,
        retrieval_mode=retrieval_mode,
    )
    current_state.retrieved_contexts = raw_contexts

    await state_queue.put(
        BackToFrontData(
            step=ExecutionState.QA_ANSWERING,
            state="citation_chunks",
            data=citation_chunks,
        )
    )

    if not context_str:
        context_str = "未检索到与该问题相关的文献片段。"

    # ---- 组装对话历史（长度受 qa.max_history_* 限制）----
    history_prompt = ""
    hm = config.get_int("qa.max_history_messages", 4)
    max_line = config.get_int("qa.max_history_message_chars", 1500)
    if current_state.chat_history:
        history_prompt = "【历史对话记录】\n"
        for msg in current_state.chat_history[-hm:]:
            role_name = "用户" if msg.get("role") == "user" else "AI助手"
            line = _truncate_text(str(msg.get("content") or ""), max_line, "历史一条")
            history_prompt += f"{role_name}：{line}\n"
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
        logger.info(
            "[QA] GraphRAG 补充上下文 | search_type=%s | 长度=%s 字符 | db_ids=%s",
            search_type,
            len(graph_context or ""),
            preferred_db_ids,
        )

        context_str, graph_context, history_prompt = _apply_qa_prompt_limits(
            context_str, graph_context, history_prompt
        )

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
        context_str, _, history_prompt = _apply_qa_prompt_limits(
            context_str, "", history_prompt
        )
        task_prompt = (
            f"【检索到的文献片段】\n{context_str}\n\n"
            f"====================\n\n"
            f"{history_prompt}"
            f"【当前用户问题】\n{user_question}"
        )
        active_agent = _make_qa_agent()

    await state_queue.put(
        BackToFrontData(step=ExecutionState.QA_ANSWERING, state="generating", data="正在生成回答...")
    )
    try:
        final_answer = await stream_assistant_to_queue(
            active_agent,
            task_prompt,
            state_queue,
            step=ExecutionState.QA_ANSWERING,
        )
    except Exception as e:
        logger.error(f"问答生成失败: {str(e)}")
        final_answer = f"抱歉，生成回答时出现错误：{str(e)}"
        current_state.error.qa_node_error = str(e)

    current_state.qa_answer = final_answer
    await state_queue.put(
        BackToFrontData(step=ExecutionState.QA_ANSWERING, state="completed", data=final_answer)
    )
    return {"value": current_state}
