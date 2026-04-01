"""AssistantAgent 流式输出：将 token 增量推入 state_queue（SSE）。"""

from __future__ import annotations

from typing import Any

from src.core.state_models import BackToFrontData, ExecutionState


def _extract_streaming_delta(event: Any) -> str | None:
    """
    从 run_stream 单条事件中提取文本增量。
    兼容不同 autogen 版本（类路径 / isinstance 可能不一致，故结合类型名判断）。
    """
    try:
        from autogen_agentchat.messages import ModelClientStreamingChunkEvent
    except ImportError:
        ModelClientStreamingChunkEvent = None  # type: ignore[misc, assignment]

    if ModelClientStreamingChunkEvent is not None and isinstance(
        event, ModelClientStreamingChunkEvent
    ):
        raw = getattr(event, "content", None)
        if isinstance(raw, str) and raw:
            return raw
        return None

    tn = getattr(type(event), "__name__", "")
    if "StreamingChunk" in tn or tn == "ModelClientStreamingChunkEvent":
        raw = getattr(event, "content", None)
        if isinstance(raw, str) and raw:
            return raw
    return None


async def stream_assistant_to_queue(
    agent: Any,
    task: str,
    state_queue: Any,
    *,
    step: str = ExecutionState.QA_ANSWERING,
) -> str:
    """
    使用 run_stream 将模型输出增量写入队列（state=\"stream_delta\"），
    返回完整文本供写入 state.qa_answer。

    若运行环境不支持 run_stream，则回退为一次性 run。
    """
    if not hasattr(agent, "run_stream"):
        result = await agent.run(task=task)
        return str(result.messages[-1].content)

    accumulated = ""
    last_from_messages = ""

    async for event in agent.run_stream(task=task):
        delta = _extract_streaming_delta(event)
        if delta is not None:
            accumulated += delta
            await state_queue.put(
                BackToFrontData(step=step, state="stream_delta", data=delta)
            )
            continue

        cm = getattr(event, "chat_message", None)
        if cm is not None:
            c = getattr(cm, "content", None)
            if isinstance(c, str) and c.strip():
                last_from_messages = c

        messages = getattr(event, "messages", None)
        if messages:
            last = messages[-1]
            c = getattr(last, "content", None)
            if c is not None:
                last_from_messages = str(c)

    if accumulated.strip():
        return accumulated
    if last_from_messages.strip():
        return last_from_messages

    result = await agent.run(task=task)
    return str(result.messages[-1].content)
