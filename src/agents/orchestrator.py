import sys
import os

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import TypedDict, Annotated, Sequence, Dict, Any
from langgraph.graph import StateGraph, END

from src.core.state_models import PaperAgentState, ExecutionState, NodeError, BackToFrontData, State, ConfigSchema
from src.agents.search_agent import search_node
from src.agents.reading_agent import reading_node
from src.agents.qa_agent import qa_node  # 引入我们刚写的 RAG 问答节点
from src.agents.intent_agent import intent_node
from src.agents.chat_agent import chat_node
import asyncio


def intent_router(state: State) -> str:
    """意图路由：多轮追问直跳 qa；否则按意图分流。"""
    current_state = state["value"]
    cfg = getattr(current_state, "config", None) or {}
    if cfg.get("bypass_to_qa"):
        return "qa_node"
    if cfg.get("intent_route") == "chat":
        return "chat_node"
    return "search_node"


class PaperAgentOrchestrator:
    def __init__(self, state_queue):
        self.state_queue = state_queue
        self.graph = self._build_graph()

    async def handle_error_node(self, state: State) -> State:
        """错误处理节点"""
        current_state = state["value"]
        current_state.current_step = ExecutionState.FAILED
        print(f"Workflow failed at {current_state.current_step}: {current_state.error}")
        return {"value": current_state}

    def condition_handler(self, state: State) -> str:
        """【修改核心】路由分发逻辑：根据当前步骤决定下一个节点"""
        current_state = state["value"]
        err = current_state.error
        current_step = current_state.current_step

        # 1. 搜索完毕 -> 进入阅读与切片建库
        if err.search_node_error is None and current_step == ExecutionState.SEARCHING:
            return "reading_node"

        # 2. 建库完毕 -> 越过复杂的分析和写作，直接进入问答节点
        elif err.reading_node_error is None and current_step == ExecutionState.READING:
            return "qa_node"

        # 3. 问答完毕 -> 结束当前执行流 (挂起，等待前端发起下一轮追问)
        elif getattr(err, 'qa_node_error', None) is None and current_step == ExecutionState.QA_ANSWERING:
            return END

        else:
            return "handle_error_node"

    def _build_graph(self):
        """构建 LangGraph：意图识别 → 闲聊 或 检索→阅读→问答"""
        builder = StateGraph(State, context_schema=ConfigSchema)

        builder.add_node("intent_node", intent_node)
        builder.add_node("search_node", search_node)
        builder.add_node("reading_node", reading_node)
        builder.add_node("qa_node", qa_node)
        builder.add_node("chat_node", chat_node)
        builder.add_node("handle_error_node", self.handle_error_node)

        builder.set_entry_point("intent_node")

        builder.add_conditional_edges("intent_node", intent_router)
        builder.add_conditional_edges("search_node", self.condition_handler)
        builder.add_conditional_edges("reading_node", self.condition_handler)
        builder.add_conditional_edges("qa_node", self.condition_handler)
        builder.add_conditional_edges("chat_node", self.condition_handler)
        builder.add_edge("handle_error_node", END)

        return builder.compile()

    async def run(
        self,
        user_request: str,
        max_papers: int = 50,
        enable_web_search: bool = True,
        retrieval_mode: str = "rag",
        selected_db_ids: list[str] | None = None,
        auto_selected_db_ids: list[str] | None = None,
    ):
        print("Starting RAG QA workflow...")
        initial_state = PaperAgentState(
            user_request=user_request,
            current_question=user_request,
            max_papers=max_papers,
            error=NodeError(),
            config={
                "enable_web_search": enable_web_search,
                "retrieval_mode": retrieval_mode,
                "selected_db_ids": selected_db_ids or [],
                "auto_selected_db_ids": auto_selected_db_ids or [],
            }
        )

        # 拿到图执行完毕后的结果
        result = await self.graph.ainvoke({"state_queue": self.state_queue, "value": initial_state})
        final_state = result["value"]
        # 首轮会话未经过 ask_question 时，补齐 assistant 消息，便于前端展示历史
        ans = getattr(final_state, "qa_answer", None)
        if ans and (
            not final_state.chat_history
            or final_state.chat_history[-1].get("role") != "assistant"
        ):
            final_state.chat_history.append({"role": "assistant", "content": ans})

        await self.state_queue.put(BackToFrontData(step=ExecutionState.FINISHED, state="finished", data=None))

        # 【新增这行】返回最终状态，供 main.py 保存到 session 中
        return final_state

    async def ask_question(
        self,
        current_state: PaperAgentState,
        new_question: str,
        enable_web_search: bool = True,
        retrieval_mode: str = "rag",
        selected_db_ids: list[str] | None = None,
        auto_selected_db_ids: list[str] | None = None,
    ):
        """
        【新增】多轮对话入口。
        当临时知识库已经建好后，前端接口直接调用此方法，跳过耗时的检索和阅读建库，直接进行极速 QA 回答。
        """
        print(f"Answering follow-up question: {new_question}")

        # 1. 变更当前问题
        current_state.current_question = new_question

        # 2. 将状态重置为“刚读完文献”的状态，便于 condition_handler 在直跳时进入 qa_node
        current_state.current_step = ExecutionState.READING

        # 3. 记录用户的追问到历史对话
        current_state.chat_history.append({"role": "user", "content": new_question})

        # 3.1 根据前端开关更新配置（是否进行联网搜索）
        current_state.config = current_state.config or {}
        current_state.config["enable_web_search"] = enable_web_search
        current_state.config["retrieval_mode"] = retrieval_mode
        if selected_db_ids is not None:
            current_state.config["selected_db_ids"] = selected_db_ids
        if auto_selected_db_ids is not None:
            current_state.config["auto_selected_db_ids"] = auto_selected_db_ids
        # 跳过多轮时的意图识别与检索，直接进入 qa_node（与原先「仅问答」行为一致）
        current_state.config["bypass_to_qa"] = True

        # 4. 再次触发工作流（intent_node 识别 bypass → qa_node）
        result = await self.graph.ainvoke({"state_queue": self.state_queue, "value": current_state})

        # 5. 将 AI 的回答追加到历史对话中，形成记忆闭环
        final_state = result["value"]
        final_state.chat_history.append({"role": "assistant", "content": final_state.qa_answer})

        await self.state_queue.put(BackToFrontData(step=ExecutionState.FINISHED, state="finished", data=None))
        return final_state


if __name__ == "__main__":
    import asyncio


    # 模拟前端的 Queue 用于本地控制台测试
    class MockQueue:
        async def put(self, item):
            print(f"[Queue] Step: {item.step}, State: {item.state}, Data: {str(item.data)[:50]}...")
            if item.state == "completed" and item.step == ExecutionState.QA_ANSWERING:
                print(f"\n======== AI 回答 ========\n{item.data}\n=========================\n")


    async def test():
        orchestrator = PaperAgentOrchestrator(MockQueue())
        # 1. 测试初次运行 (建库 + 回答)
        print(">>> 模拟第一次提问（包含建库）")
        await orchestrator.run("帮我找几篇关于大模型在自动驾驶中应用的最新论文，并总结它们的传感器融合方案。")

        # 2. 测试多轮追问 (仅检索不重新建库)
        # 注意：在真实 FastAPI 环境中，你会从内存或 Redis 里把刚刚完成的 final_state 捞出来传给 ask_question
        # 这里仅为逻辑展示，因为独立测试脚本跑完就释放了


    asyncio.run(test())