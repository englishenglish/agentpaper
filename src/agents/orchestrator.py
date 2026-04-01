from langgraph.graph import StateGraph, END

from src.core.state_models import PaperAgentState, ExecutionState, NodeError, BackToFrontData, State, ConfigSchema, MAX_CHAT_HISTORY
from src.agents.search_agent import search_node
from src.agents.reading_agent import reading_node
from src.agents.qa_agent import qa_node
from src.agents.intent_agent import intent_node
from src.agents.chat_agent import chat_node


def intent_router(state: State) -> str:
    """意图路由：闲聊 / 直连问答（已选手动库）/ 联网建库全流程。"""
    current_state = state["value"]
    cfg = getattr(current_state, "config", None) or {}

    intent = cfg.get("intent_route", "research")
    has_db = cfg.get("bypass_to_qa", False)
    kb_binding = cfg.get("kb_binding")
    selected = cfg.get("selected_db_ids") or []

    if intent == "chat":
        return "chat_node"
    if intent == "research":
        if has_db:
            return "qa_node"
        # 手动选了知识库：不联网搜论文，直接基于该库问答
        if kb_binding == "manual" and selected:
            return "qa_node"
        return "search_node"
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
        """路由分发逻辑：根据当前步骤决定下一个节点"""
        current_state = state["value"]
        err = current_state.error
        current_step = current_state.current_step

        # 1. 搜索完毕 -> 进入阅读与切片建库
        if err.search_node_error is None and current_step == ExecutionState.SEARCHING:
            return "reading_node"

        # 2. 建库完毕 -> 越过复杂的分析和写作，直接进入问答节点
        elif err.reading_node_error is None and current_step == ExecutionState.READING:
            return "qa_node"

        # 3. 问答完毕 -> 结束当前执行流，并打上“已建库”标记
        elif err.qa_node_error is None and current_step == ExecutionState.QA_ANSWERING:
            if current_state.config is None:
                current_state.config = {}
            current_state.config["bypass_to_qa"] = True  # 打上建库成功的思想钢印
            return END

        else:
            return "handle_error_node"

    def chat_condition_handler(self, state: State) -> str:
        """处理闲聊节点的安全退出"""
        current_state = state["value"]
        if current_state.error.chat_node_error is None:
            return END
        return "handle_error_node"

    def _build_graph(self):
        """构建 LangGraph"""
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
        # 绑定专属的 chat 退出处理器
        builder.add_conditional_edges("chat_node", self.chat_condition_handler)
        builder.add_edge("handle_error_node", END)

        return builder.compile()

    async def run(
            self,
            user_request: str,
            previous_state: PaperAgentState = None,  # 【新增参数】接收历史对话状态
            max_papers: int = 50,
            enable_web_search: bool = True,
            retrieval_mode: str = "rag",
            selected_db_ids: list[str] | None = None,
    ):
        print("Starting Paper/Chat workflow...")

        sel = list(selected_db_ids or [])

        # 【修改逻辑】如果有历史状态，则继承历史并更新问题；否则从零创建
        if previous_state:
            initial_state = previous_state
            initial_state.current_question = user_request
            initial_state.chat_history.append({"role": "user", "content": user_request})
            if len(initial_state.chat_history) > MAX_CHAT_HISTORY:
                initial_state.chat_history = initial_state.chat_history[-MAX_CHAT_HISTORY:]
            initial_state.error = NodeError()
            if initial_state.config is None:
                initial_state.config = {}
            cfg = initial_state.config
            kb_binding = cfg.get("kb_binding")
            # 已绑定手动库或联网创建的库：锁定该会话知识库，且不再联网搜索
            if kb_binding in ("manual", "built"):
                cfg["enable_web_search"] = False
                locked = cfg.get("selected_db_ids") or []
                cfg["selected_db_ids"] = list(locked) if locked else []
            else:
                cfg["enable_web_search"] = enable_web_search
                cfg["selected_db_ids"] = sel
            cfg["retrieval_mode"] = retrieval_mode
        else:
            cfg = {
                "retrieval_mode": retrieval_mode,
                "bypass_to_qa": False,
                "selected_db_ids": sel,
                "enable_web_search": enable_web_search if not sel else False,
            }
            if sel:
                cfg["kb_binding"] = "manual"

            initial_state = PaperAgentState(
                user_request=user_request,
                current_question=user_request,
                max_papers=max_papers,
                error=NodeError(),
                config=cfg,
            )
            initial_state.chat_history.append({"role": "user", "content": user_request})

        # 触发图运行
        result = await self.graph.ainvoke({"state_queue": self.state_queue, "value": initial_state})
        final_state = result["value"]

        # 补齐 assistant 消息，便于前端展示历史
        ans = getattr(final_state, "qa_answer", None) or getattr(final_state, "chat_answer", None)
        if ans and (
                not final_state.chat_history
                or final_state.chat_history[-1].get("role") != "assistant"
        ):
            final_state.chat_history.append({"role": "assistant", "content": ans})

        await self.state_queue.put(BackToFrontData(step=ExecutionState.FINISHED, state="finished", data=None))

        # 返回最终状态，供外部缓存，下一轮对话时当作 previous_state 传回来
        return final_state

    # 【注意】原先的 ask_question 方法已被安全删除，所有逻辑被无缝融入了 run() 中。


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