from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from src.agents.userproxy_agent import WebUserProxyAgent, userProxyAgent
from pydantic import BaseModel, Field
from typing import Optional, List
import re
import ast
import asyncio
from openai import RateLimitError
from src.utils.log_utils import setup_logger
from src.tasks.paper_search import PaperSearcher
from src.core.state_models import State, ExecutionState
from src.core.prompts import search_agent_prompt
from src.core.state_models import BackToFrontData

from src.core.model_client import create_search_model_client

logger = setup_logger(__name__)

model_client = create_search_model_client()


# 创建一个查询条件类，包括查询内容、主题、时间范围等信息，用于存储用户的查询需求
class SearchQuery(BaseModel):
    """查询条件类，存储用户查询需求"""
    querys: Optional[List[str]] = Field(default=None, description="查询条件列表")
    start_date: Optional[str] = Field(default=None, description="开始时间, 格式: YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="结束时间, 格式: YYYY-MM-DD")


def _make_search_agent() -> AssistantAgent:
    """每次调用创建新实例，避免多并发请求共享对话历史导致上下文污染"""
    return AssistantAgent(
        name="search_agent",
        model_client=model_client,
        system_message=search_agent_prompt,
        output_content_type=SearchQuery,
    )



async def search_node(state: State) -> State:
    """搜索论文节点"""
    state_queue = None
    try:
        state_queue = state["state_queue"]
        current_state = state["value"]
        current_state.current_step = ExecutionState.SEARCHING
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="initializing", data=None))

        # 根据 config 中的 enable_web_search 决定是否执行联网检索
        enable_web_search = True
        try:
            cfg = getattr(current_state, "config", {}) or {}
            enable_web_search = bool(cfg.get("enable_web_search", True))
        except Exception:
            enable_web_search = True

        if not enable_web_search:
            msg = "已关闭联网搜索，跳过 arXiv 文献检索，将基于现有知识库（若已配置）进行问答。"
            logger.info(msg)
            current_state.search_results = []
            await state_queue.put(
                BackToFrontData(
                    step=ExecutionState.SEARCHING,
                    state="skipped",
                    data=msg,
                )
            )
            await state_queue.put(
                BackToFrontData(
                    step=ExecutionState.SEARCHING,
                    state="completed",
                    data="联网检索已跳过。",
                )
            )
            return {"value": current_state}
        # 1. 生成检索条件
        prompt = f"""
        请根据用户查询需求，生成检索查询条件。
        用户查询需求：{current_state.user_request}
        """
        response = await safe_llm_call(_make_search_agent(), prompt)
        parsed_query: SearchQuery = response.messages[-1].content
        logger.info(f"Agent 生成的查询条件对象：{parsed_query}")

        # 2. 解析为 SearchQuery 对象（核心：确保解析后是合法的对象）
        # parsed_query = parse_search_query(search_query_str)
        if not parsed_query.querys or len(parsed_query.querys) == 0:
            err_msg = "生成的查询条件为空，请重新输入查询需求"
            current_state.error.search_node_error = err_msg
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=err_msg))
            return {"value": current_state}

        # 3. 通知前端人工审核（如果你需要保留人工审核逻辑，取消下面注释）
        # await state_queue.put(
        #     BackToFrontData(step=ExecutionState.SEARCHING, state="user_review", data=f"请审核查询条件：{search_query_str}")
        # )
        # # 等待人工确认
        # result = await userProxyAgent.on_messages(
        #     [TextMessage(content=f"请人工审核以下查询条件是否符合要求：\n{search_query_str}\n回复'确认'或修改后的条件", source="AI")],
        #     cancellation_token=CancellationToken()
        # )
        # # 重新解析人工确认/修改后的条件
        # parsed_query = parse_search_query(result.content)

        # 4. 转换查询条件为字符串（核心修复：满足 search_papers 入参要求）
        combined_query = " ".join(parsed_query.querys)

        # 5. 调用检索服务
        await state_queue.put(BackToFrontData(
            step=ExecutionState.SEARCHING,
            state="processing",
            data=f"正在检索关键词：{combined_query}（时间范围：{parsed_query.start_date or '不限'} 至 {parsed_query.end_date or '不限'}）..."
        ))

        paper_searcher = PaperSearcher()

        results = await paper_searcher.search_papers(
            querys=parsed_query.querys,
            start_date=parsed_query.start_date,
            end_date=parsed_query.end_date,
        )

        # 6. 处理检索结果
        current_state.search_results = results
        if len(results) > 0:
            await state_queue.put(BackToFrontData(
                step=ExecutionState.SEARCHING,
                state="completed",
                data=f"文献检索完成，共获取 {len(results)} 篇论文，即将开始构建问答知识库..."
            ))
        else:
            err_msg = "没有找到相关论文，请尝试调整关键词或时间范围"
            current_state.error.search_node_error = err_msg
            await state_queue.put(BackToFrontData(
                step=ExecutionState.SEARCHING,
                state="error",
                data=err_msg
            ))

        return {"value": current_state}

    except Exception as e:
        err_msg = f"Search failed: {str(e)}"
        logger.error(f"搜索节点执行失败：{err_msg}", exc_info=True)
        # 确保 state_queue 不为空时才发送消息
        if state_queue:
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=err_msg))
        # 记录错误到 state
        if "value" in state and hasattr(state["value"], "error"):
            state["value"].error.search_node_error = err_msg
        return state


async def safe_llm_call(agent, prompt, retries=3):
    import asyncio
    from openai import RateLimitError

    for i in range(retries):
        try:
            return await agent.run(task=prompt)
        except RateLimitError:
            wait = 5 * (i + 1)
            logger.warning(f"Rate limit hit, retrying in {wait}s...")
            await asyncio.sleep(wait)

    raise Exception("LLM rate limit exceeded")

