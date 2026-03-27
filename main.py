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
from src.knowledge.router import knowledge
from src.knowledge import knowledge_base
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
    """接口 1：初始化知识库（其实现在和 /chat 逻辑几乎一致，保留独立入口方便前端语义化）"""
    print("API /api/research/init called")
    if not session_id:
        session_id = str(uuid.uuid4())

    session_queue = asyncio.Queue()
    active_sessions[session_id] = {"queue": session_queue, "state": None, "ts": time.time()}
    _cleanup_expired_sessions()

    manual_selected = _parse_selected_db_ids(selected_db_ids)
    auto_selected = [] if manual_selected else _auto_select_knowledge_bases(query, top_k=3)

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
                enable_web_search=enable_web_search,
                retrieval_mode=retrieval_mode,
                selected_db_ids=manual_selected,
                auto_selected_db_ids=auto_selected,
            )
            active_sessions[session_id]["state"] = final_state
        except Exception as e:
            logger.error(f"Init workflow failed: {e}")
            await session_queue.put(
                BackToFrontData(step=ExecutionState.FINISHED, state="error", data=str(e))
            )

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
    """接口 2：多轮对话追问。完全信任编排器的路由决策。"""
    chat_queue = asyncio.Queue()

    if session_id not in active_sessions:
        active_sessions[session_id] = {"queue": chat_queue, "state": None, "ts": time.time()}
    else:
        active_sessions[session_id]["queue"] = chat_queue
        active_sessions[session_id]["ts"] = time.time()

    current_state = active_sessions[session_id]["state"]
    manual_selected = _parse_selected_db_ids(selected_db_ids)
    auto_selected = [] if manual_selected else _auto_select_knowledge_bases(question, top_k=3)

    async def event_generator():
        while True:
            state_data = await chat_queue.get()
            yield {"data": state_data.model_dump_json()}
            if state_data.step == ExecutionState.FINISHED:
                break

    async def run_chat_task():
        orchestrator = PaperAgentOrchestrator(state_queue=chat_queue)
        try:
            # 如果是已有会话，前端可能改变了开关配置，我们在传入前更新一下 config
            if current_state and current_state.config:
                current_state.config["enable_web_search"] = enable_web_search
                current_state.config["retrieval_mode"] = retrieval_mode
                if manual_selected:
                    current_state.config["selected_db_ids"] = manual_selected
                if auto_selected:
                    current_state.config["auto_selected_db_ids"] = auto_selected

            # 【核心修改：极简调用】
            # 不再做繁琐的 has_any_db 判断，直接把 current_state 丢进去，
            # 编排器会根据意图和 bypass_to_qa 自己决定该走哪条路！
            final_state = await orchestrator.run(
                user_request=question,
                previous_state=current_state,
                enable_web_search=enable_web_search,
                retrieval_mode=retrieval_mode,
                selected_db_ids=manual_selected,
                auto_selected_db_ids=auto_selected,
            )
            active_sessions[session_id]["state"] = final_state
        except Exception as e:
            logger.error(f"Chat workflow failed: {e}")
            await chat_queue.put(
                BackToFrontData(step=ExecutionState.FINISHED, state="error", data=str(e))
            )

    asyncio.create_task(run_chat_task())
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