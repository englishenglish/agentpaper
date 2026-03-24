import asyncio
import time
import uuid
import re
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk
from src.agents.userproxy_agent import WebUserProxyAgent, userProxyAgent
from src.knowledge.knowledge_router import knowledge
from src.knowledge.knowledge import knowledge_base
from src.core.state_models import BackToFrontData, ExecutionState
from src.agents.orchestrator import PaperAgentOrchestrator

# 设置日志
logger = setup_logger(name='main', log_file='project.log')

app = FastAPI()
app.include_router(knowledge)

# === CORS 配置 ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 全局会话管理器 (Session Store)
# 格式: { "session_id": {"queue": asyncio.Queue, "state": PaperAgentState, "ts": float} }
# ==========================================
active_sessions: dict = {}
_SESSION_TTL = 3600  # 会话最长保留 1 小时


def _cleanup_expired_sessions() -> None:
    """清理超过 TTL 的过期会话，防止内存泄漏"""
    now = time.time()
    expired = [sid for sid, s in active_sessions.items() if now - s.get("ts", 0) > _SESSION_TTL]
    for sid in expired:
        active_sessions.pop(sid, None)
    if expired:
        logger.info(f"清理过期会话 {len(expired)} 个")


def _parse_selected_db_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [x.strip() for x in raw.split(",")]
    return [x for x in parts if x]


def _tokenize_query(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower())


def _auto_select_knowledge_bases(question: str, top_k: int = 3) -> list[str]:
    """
    在用户未手动选择知识库时，自动挑选最相关的几个库。
    """
    try:
        dbs = knowledge_base.get_databases().get("databases", [])
    except Exception:
        return []

    keywords = _tokenize_query(question)
    if not keywords:
        return []

    scored = []
    for db in dbs:
        db_id = db.get("db_id")
        if not db_id:
            continue
        text = f"{db.get('name', '')} {db.get('description', '')}".lower()
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        if score > 0:
            scored.append((score, db_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [db_id for _, db_id in scored[:top_k]]


@app.post("/send_input")
async def send_input(data: dict):
    """人类在环（人工审核）的输入接口保持不变"""
    user_input = data.get("input")
    userProxyAgent.set_user_input(user_input)
    return JSONResponse({"status": 200, "msg": "已收到人工输入"})


@app.get('/api/research/init')
async def research_init_stream(
    query: str,
    session_id: str = Query(default=None),
    enable_web_search: bool = Query(default=True),
    retrieval_mode: str = Query(default="rag"),
    selected_db_ids: str | None = Query(default=None),
):
    """
    接口 1：初始化知识库并生成首次问答。
    包含：检索 -> 阅读切片入库 -> 首次 QA 生成
    """
    print("API /api/research/init called")
    if not session_id:
        session_id = str(uuid.uuid4())  # 没传则生成一个唯一的会话 ID
    print(3)
    # 为当前会话创建专属的通信队列，并记录时间戳
    session_queue = asyncio.Queue()
    active_sessions[session_id] = {"queue": session_queue, "state": None, "ts": time.time()}
    _cleanup_expired_sessions()

    manual_selected = _parse_selected_db_ids(selected_db_ids)
    auto_selected = [] if manual_selected else _auto_select_knowledge_bases(query, top_k=3)
    matched_db_ids = manual_selected or auto_selected

    async def event_generator():
        print("SSE generator start")
        yield {"data": BackToFrontData(step="system", state="session_created",
                                       data={"session_id": session_id}).model_dump_json()}

        while True:
            print("waiting queue...")
            state_data = await session_queue.get()
            print("got queue:", state_data)
            yield {"data": state_data.model_dump_json()}
            if state_data.step == ExecutionState.FINISHED:
                break

    async def run_task():
        orchestrator = PaperAgentOrchestrator(state_queue=session_queue)
        try:
            # 命中已有知识库时，优先走本地库问答（无需联网搜索）
            final_enable_web_search = enable_web_search if not matched_db_ids else False
            # 运行全流程，并拿回包含知识库 ID 和历史记录的最终状态
            final_state = await orchestrator.run(
                user_request=query,
                enable_web_search=final_enable_web_search,
                retrieval_mode=retrieval_mode,
                selected_db_ids=manual_selected,
                auto_selected_db_ids=auto_selected,
            )
            # 存入字典，供多轮聊天使用
            active_sessions[session_id]["state"] = final_state
        except Exception as e:
            logger.error(f"Init workflow failed: {e}")
            await session_queue.put(
                BackToFrontData(step=ExecutionState.FINISHED, state="error", data=str(e))
            )

    # 启动后台异步任务
    asyncio.create_task(run_task())

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


@app.get('/api/research/chat')
async def research_chat_stream(
    question: str,
    session_id: str,
    enable_web_search: bool = Query(default=True),
    retrieval_mode: str = Query(default="rag"),
    selected_db_ids: str | None = Query(default=None),
):
    """
    接口 2：多轮问答追问接口。
    如果会话不存在，则自动按 /init 流程创建会话并完成首次问答（向后端“自恢复”），
    这样前端可以直接使用 /chat 作为统一入口。
    """
    # 每次请求创建新队列，避免读到上次请求的残留消息
    chat_queue = asyncio.Queue()
    # 如果会话不存在，初始化一条空状态，行为等价于 /init
    if session_id not in active_sessions:
        active_sessions[session_id] = {"queue": chat_queue, "state": None, "ts": time.time()}
    else:
        active_sessions[session_id]["queue"] = chat_queue
        active_sessions[session_id]["ts"] = time.time()

    # 取当前会话状态（可能为 None，代表还未执行过完整流程）
    current_state = active_sessions[session_id]["state"]

    manual_selected = _parse_selected_db_ids(selected_db_ids)
    auto_selected = [] if manual_selected else _auto_select_knowledge_bases(question, top_k=3)
    matched_db_ids = manual_selected or auto_selected

    async def event_generator():
        while True:
            state_data = await chat_queue.get()
            yield {"data": state_data.model_dump_json()}
            if state_data.step == ExecutionState.FINISHED:
                break

    async def run_chat_task():
        orchestrator = PaperAgentOrchestrator(state_queue=chat_queue)
        try:
            # 全新会话（state 为空），走完整流程：意图→搜索→建库→QA
            if current_state is None:
                final_state = await orchestrator.run(
                    user_request=question,
                    enable_web_search=enable_web_search,
                    retrieval_mode=retrieval_mode,
                    selected_db_ids=manual_selected,
                    auto_selected_db_ids=auto_selected,
                )
                active_sessions[session_id]["state"] = final_state
                return

            # ---- 已有会话：追问逻辑 ----
            # 优先使用用户手动选择的库；其次复用上轮对话中已建好的临时库；
            # 最后才降级到本轮 auto_select（避免因措辞变化重复联网建库）
            session_cfg = getattr(current_state, "config", {}) or {}
            session_auto_db_ids = (
                session_cfg.get("auto_selected_db_ids")
                or session_cfg.get("selected_db_ids")
                or []
            )
            effective_auto = manual_selected or session_auto_db_ids or auto_selected
            effective_manual = manual_selected

            # 只有当用户明确开启联网 且 本轮 + session 都找不到任何知识库时，才重新建库
            has_any_db = bool(effective_auto or effective_manual)
            if not has_any_db and enable_web_search:
                final_state = await orchestrator.run(
                    user_request=question,
                    enable_web_search=True,
                    retrieval_mode=retrieval_mode,
                    selected_db_ids=[],
                    auto_selected_db_ids=[],
                )
                active_sessions[session_id]["state"] = final_state
                return

            # 直接追问：跳过搜索/建库，基于已有知识库回答
            final_state = await orchestrator.ask_question(
                current_state=current_state,
                new_question=question,
                enable_web_search=False,  # 追问阶段不联网，始终基于已建知识库
                retrieval_mode=retrieval_mode,
                selected_db_ids=effective_manual,
                auto_selected_db_ids=effective_auto,
            )
            active_sessions[session_id]["state"] = final_state
        except Exception as e:
            logger.error(f"Chat workflow failed: {e}")
            await chat_queue.put(
                BackToFrontData(step=ExecutionState.FINISHED, state="error", data=str(e))
            )

    # 启动后台聊天任务
    asyncio.create_task(run_chat_task())

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    """
    健康检查与简要运行状态：
    - 后端服务存活状态
    - 当前活跃会话数量
    - 知识库基础统计信息
    """
    kb_stats = {}
    try:
        kb_stats = knowledge_base.get_statistics()
    except Exception as e:
        logger.error(f"获取知识库统计信息失败: {e}")

    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "knowledge_base": kb_stats,
    }


if __name__ == "__main__":
    import uvicorn

    # 为了防止因为开发热更新导致内存中的 active_sessions 丢失，生产环境建议配合 Redis 存储状态
    uvicorn.run(app, host="0.0.0.0", port=8000)