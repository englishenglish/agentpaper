"""意图识别节点：分流「普通闲聊」与「文献检索 + RAG」主流程。"""

from __future__ import annotations

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autogen_agentchat.agents import AssistantAgent

from src.core.model_client import create_search_model_client
from src.core.prompts import intent_classifier_prompt
from src.core.state_models import State, ExecutionState, BackToFrontData
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_intent_client = create_search_model_client()


def _make_intent_agent() -> AssistantAgent:
    return AssistantAgent(
        name="intent_agent",
        model_client=_intent_client,
        system_message=intent_classifier_prompt,
    )


def _parse_intent(raw: str) -> str:
    """从模型的文本输出中提取 chat / research。默认 research。"""
    text = raw.strip().lower()
    if "chat" in text and "research" not in text:
        return "chat"
    return "research"


async def intent_node(state: State) -> State:
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.INTENT_RECOGNIZING

    # 【修改：删除了 bypass_to_qa 的拦截逻辑，强制每次都进行大模型意图识别】

    user_text = (current_state.current_question or current_state.user_request or "").strip()
    await state_queue.put(
        BackToFrontData(
            step=ExecutionState.INTENT_RECOGNIZING,
            state="processing",
            data="正在识别对话意图...",
        )
    )

    try:
        prompt = f"用户输入：\n{user_text}\n\n请判断 intent（只输出 chat 或 research，不要输出其他任何内容）。"
        result = await _make_intent_agent().run(task=prompt)
        raw_content = result.messages[-1].content
        route = _parse_intent(str(raw_content))
    except Exception as e:
        logger.warning(f"意图识别失败，默认走 research: {e}")
        route = "research"

    if current_state.config is None:
        current_state.config = {}

    # 记录真正的意图
    current_state.config["intent_route"] = route

    await state_queue.put(
        BackToFrontData(
            step=ExecutionState.INTENT_RECOGNIZING,
            state="completed",
            data="闲聊模式" if route == "chat" else "文献检索与学术问答",
        )
    )
    return {"value": current_state}