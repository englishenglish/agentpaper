"""普通闲聊节点：不检索、不建库，直接由 LLM 生成回复。"""

from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent

from src.core.model_client import create_qa_model_client
from src.core.prompts import casual_chat_agent_prompt
from src.core.state_models import State, ExecutionState, BackToFrontData
from src.utils.log_utils import setup_logger
from src.agents.streaming_utils import stream_assistant_to_queue

logger = setup_logger(__name__)

_chat_model_client = None


def _get_chat_model_client():
    global _chat_model_client
    if _chat_model_client is None:
        _chat_model_client = create_qa_model_client()
    return _chat_model_client


def _make_chat_agent() -> AssistantAgent:
    return AssistantAgent(
        name="chat_agent",
        model_client=_get_chat_model_client(),
        system_message=casual_chat_agent_prompt,
        model_client_stream=True,
    )


async def chat_node(state: State) -> State:
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.QA_ANSWERING

    user_q = (current_state.current_question or current_state.user_request or "").strip()

    history_prompt = ""
    if current_state.chat_history:
        history_prompt = "【历史对话】\n"
        for msg in current_state.chat_history[-6:]:
            role_name = "用户" if msg.get("role") == "user" else "助手"
            history_prompt += f"{role_name}：{msg.get('content', '')}\n"
        history_prompt += "\n"

    if (
        not current_state.chat_history
        or current_state.chat_history[-1].get("role") != "user"
        or current_state.chat_history[-1].get("content") != user_q
    ):
        current_state.chat_history.append({"role": "user", "content": user_q})

    task_prompt = f"{history_prompt}【用户当前消息】\n{user_q}"

    await state_queue.put(
        BackToFrontData(step=ExecutionState.QA_ANSWERING, state="status", data="正在回复...")
    )
    try:
        await state_queue.put(
            BackToFrontData(step=ExecutionState.QA_ANSWERING, state="generating", data="正在生成回复...")
        )
        final_answer = await stream_assistant_to_queue(
            _make_chat_agent(),
            task_prompt,
            state_queue,
            step=ExecutionState.QA_ANSWERING,
        )
    except Exception as e:
        logger.error(f"闲聊生成失败: {e}")
        final_answer = f"抱歉，回复时出错：{e}"
        current_state.error.chat_node_error = str(e)

    current_state.qa_answer = final_answer
    await state_queue.put(
        BackToFrontData(step=ExecutionState.QA_ANSWERING, state="completed", data=final_answer)
    )
    return {"value": current_state}
