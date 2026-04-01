import asyncio
import os
import time
import uuid
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.utils.log_utils import setup_logger
from src.utils.fitz_tools import configure_fitz_mupdf_console

configure_fitz_mupdf_console()

from src.utils.tool_utils import handlerChunk
from src.agents.userproxy_agent import WebUserProxyAgent, userProxyAgent
from src.knowledge.router import knowledge
from src.knowledge import knowledge_base
from src.core.state_models import BackToFrontData, ExecutionState
from src.agents.orchestrator import PaperAgentOrchestrator

# 设置日志
logger = setup_logger(name='main', log_file='project.log')

app = FastAPI()
app.include_router(knowledge)

# === CORS 配置 ===
_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 请求安全常量 ===
_MAX_QUERY_LENGTH = 4000

# ==========================================
# 全局会话管理器 (Session Store)
# 格式: { "session_id": {"queue": asyncio.Queue, "state": PaperAgentState,
#          "ts": float, "task": asyncio.Task | None} }
# ==========================================
active_sessions: dict = {}
_SESSION_TTL = 3600


def _cleanup_expired_sessions() -> None:
    """清理超过 TTL 的过期会话，防止内存泄漏"""
    now = time.time()
    expired = [sid for sid, s in active_sessions.items() if now - s.get("ts", 0) > _SESSION_TTL]
    for sid in expired:
        active_sessions.pop(sid, None)
    if expired:
        logger.info(f"清理过期会话 {len(expired)} 个")


def _parse_single_selected_db_id(raw: str | None) -> list[str]:
    """仅接受用户手动选择的一个知识库 id；空则返回空列表（不做自动匹配）。"""
    if not raw or not str(raw).strip():
        return []
    return [str(raw).strip()]


@app.post("/send_input")
async def send_input(data: dict):
    user_input = data.get("input")
    userProxyAgent.set_user_input(user_input)
    return JSONResponse({"status": 200, "msg": "已收到人工输入"})


@app.get('/api/research/init')
async def research_init_stream(
        query: str,
        session_id: str = Query(default=None),
        enable_web_search: bool = Query(default=True),
        retrieval_mode: str = Query(default="rag"),
        selected_db_id: str | None = Query(default=None),
):
    """接口 1：初始化研究流程"""
    if not query or not query.strip():
        return JSONResponse({"error": "query 不能为空"}, status_code=400)
    if len(query) > _MAX_QUERY_LENGTH:
        return JSONResponse({"error": f"query 长度不能超过 {_MAX_QUERY_LENGTH} 字符"}, status_code=400)

    if not session_id:
        session_id = str(uuid.uuid4())

    session_queue = asyncio.Queue()
    active_sessions[session_id] = {"queue": session_queue, "state": None, "ts": time.time(), "task": None}
    _cleanup_expired_sessions()

    manual_selected = _parse_single_selected_db_id(selected_db_id)
    effective_web = False if manual_selected else enable_web_search

    async def event_generator():
        yield {"data": BackToFrontData(step="system", state="session_created",
                                       data={"session_id": session_id}).model_dump_json()}
        while True:
            state_data = await session_queue.get()
            yield {"data": state_data.model_dump_json()}
            if state_data.step == ExecutionState.FINISHED:
                break

    async def run_task():
        orchestrator = PaperAgentOrchestrator(state_queue=session_queue)
        try:
            # 【修改】使用统一的 run 方法，因为是 init，所以 previous_state 明确传 None
            final_state = await orchestrator.run(
                user_request=query,
                previous_state=None,
                enable_web_search=effective_web,
                retrieval_mode=retrieval_mode,
                selected_db_ids=manual_selected,
            )
            active_sessions[session_id]["state"] = final_state
        except Exception as e:
            logger.error(f"Init workflow failed: {e}")
            await session_queue.put(
                BackToFrontData(step=ExecutionState.FINISHED, state="error", data=str(e))
            )

    task = asyncio.create_task(run_task())
    active_sessions[session_id]["task"] = task
    return EventSourceResponse(event_generator(), media_type="text/event-stream")


@app.get('/api/research/chat')
async def research_chat_stream(
        question: str,
        session_id: str,
        enable_web_search: bool = Query(default=True),
        retrieval_mode: str = Query(default="rag"),
        selected_db_id: str | None = Query(default=None),
):
    """接口 2：多轮对话追问。"""
    if not question or not question.strip():
        return JSONResponse({"error": "question 不能为空"}, status_code=400)
    if len(question) > _MAX_QUERY_LENGTH:
        return JSONResponse({"error": f"question 长度不能超过 {_MAX_QUERY_LENGTH} 字符"}, status_code=400)

    chat_queue = asyncio.Queue()

    if session_id not in active_sessions:
        active_sessions[session_id] = {"queue": chat_queue, "state": None, "ts": time.time(), "task": None}
    else:
        prev_task = active_sessions[session_id].get("task")
        if prev_task and not prev_task.done():
            prev_task.cancel()
            logger.info(f"Cancelled previous task for session {session_id}")
        active_sessions[session_id]["queue"] = chat_queue
        active_sessions[session_id]["ts"] = time.time()

    current_state = active_sessions[session_id]["state"]
    manual_selected = _parse_single_selected_db_id(selected_db_id)
    effective_web = False if manual_selected else enable_web_search

    async def event_generator():
        while True:
            state_data = await chat_queue.get()
            yield {"data": state_data.model_dump_json()}
            if state_data.step == ExecutionState.FINISHED:
                break

    async def run_chat_task():
        orchestrator = PaperAgentOrchestrator(state_queue=chat_queue)
        try:
            # config 由 orchestrator.run 在继承 previous_state 时合并
            # 【核心修改：极简调用】
            # 不再做繁琐的 has_any_db 判断，直接把 current_state 丢进去，
            # 编排器会根据意图和 bypass_to_qa 自己决定该走哪条路！
            final_state = await orchestrator.run(
                user_request=question,
                previous_state=current_state,
                enable_web_search=effective_web,
                retrieval_mode=retrieval_mode,
                selected_db_ids=manual_selected,
            )
            active_sessions[session_id]["state"] = final_state
        except Exception as e:
            logger.error(f"Chat workflow failed: {e}")
            await chat_queue.put(
                BackToFrontData(step=ExecutionState.FINISHED, state="error", data=str(e))
            )

    task = asyncio.create_task(run_chat_task())
    active_sessions[session_id]["task"] = task
    return EventSourceResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health_check():
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