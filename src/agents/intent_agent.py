"""意图识别节点：分流「普通闲聊」与「文献检索 + RAG」主流程。"""

from __future__ import annotations

import json
import re
from typing import Any

from autogen_agentchat.agents import AssistantAgent

from src.core.model_client import create_search_model_client
from src.core.prompts import intent_classifier_prompt
from src.core.state_models import State, ExecutionState, BackToFrontData
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_intent_client = None

# 随附多轮上下文的条数上限（user/assistant 各算一条）
_INTENT_HISTORY_MAX_MESSAGES = 8
_INTENT_HISTORY_CHAR_CAP = 2400

# 解析用：JSON 中的 intent 字段
_RE_JSON_INTENT = re.compile(r'"intent"\s*:\s*"([^"]+)"', re.IGNORECASE)
# 单词级回退，避免子串误匹配（如其它词中含 chat）
_RE_WORD = re.compile(r"\b(chat|research)\b", re.IGNORECASE)

# 极短结束语：模型偶发输出非 JSON 时，保守判为闲聊（可选）
_RE_SHORT_CLOSING = re.compile(
    r"^(谢谢|感谢|多谢|辛苦了|再见|拜拜|不用了|好的谢谢|谢谢啦|明白|收到)[\s!！。.…]*$",
    re.IGNORECASE,
)


def _get_intent_client():
    global _intent_client
    if _intent_client is None:
        _intent_client = create_search_model_client()
    return _intent_client


def _make_intent_agent() -> AssistantAgent:
    return AssistantAgent(
        name="intent_agent",
        model_client=_get_intent_client(),
        system_message=intent_classifier_prompt,
    )


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = re.sub(r"^```\w*\s*", "", t, count=1)
    if t.rstrip().endswith("```"):
        t = t.rstrip()[:-3].rstrip()
    return t


def _parse_intent(raw: str) -> str:
    """
    从模型输出解析 chat / research。
    优先解析 JSON；失败则用语义边界单词；默认 research。
    """
    text = _strip_code_fence(str(raw).strip())

    # 1) 完整 JSON 对象
    if "{" in text and "}" in text:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
            v = obj.get("intent")
            if isinstance(v, str):
                v = v.strip().lower()
                if v in ("chat", "research"):
                    return v
        except (json.JSONDecodeError, ValueError):
            pass

    # 2) 正则提取 "intent":"..."
    m = _RE_JSON_INTENT.search(text)
    if m:
        v = m.group(1).strip().lower()
        if v in ("chat", "research"):
            return v

    # 3) 单词边界（若模型只输出 research 一词）
    words = _RE_WORD.findall(text.lower())
    if words:
        # 若同时出现，优先 research（与产品默认一致）
        if "research" in words:
            return "research"
        if "chat" in words:
            return "chat"

    return "research"


def _history_snippet(chat_history: list[dict[str, Any]] | None, current_question: str) -> str:
    if not chat_history or len(chat_history) < 2:
        return ""
    # 去掉与当前句完全重复的末尾用户消息，避免重复一行
    hist = chat_history[:-1] if (
        chat_history
        and chat_history[-1].get("role") == "user"
        and (chat_history[-1].get("content") or "").strip() == current_question
    ) else chat_history

    tail = hist[-_INTENT_HISTORY_MAX_MESSAGES:]
    lines: list[str] = []
    total = 0
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "助手"
        piece = f"{label}：{content}"
        if total + len(piece) > _INTENT_HISTORY_CHAR_CAP:
            break
        lines.append(piece)
        total += len(piece) + 1
    if not lines:
        return ""
    return "\n".join(lines)


def _session_hints(cfg: dict[str, Any]) -> str:
    parts: list[str] = []
    if cfg.get("bypass_to_qa"):
        parts.append("当前会话已成功建库，后续追问多与文献相关。")
    sels = cfg.get("selected_db_ids") or []
    if sels:
        parts.append("用户已选择知识库，问题更可能针对库内文献。")
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def _build_intent_task(user_text: str, cfg: dict[str, Any], chat_history: list[dict[str, Any]] | None) -> str:
    hints = _session_hints(cfg)
    hist = _history_snippet(chat_history or [], user_text)
    blocks = []
    if hints:
        blocks.append(hints.rstrip())
    if hist:
        blocks.append("【最近对话】\n" + hist)
    blocks.append("【当前用户输入】\n" + user_text)
    blocks.append("请只输出一行 JSON：{\"intent\":\"chat\"} 或 {\"intent\":\"research\"}")
    return "\n\n".join(blocks)


def _maybe_short_closing_chat(user_text: str) -> bool:
    """极短结束语，无上下文时可不走大模型（省延迟）；有学术上下文时仍走模型。"""
    t = user_text.strip()
    if len(t) > 32:
        return False
    return bool(_RE_SHORT_CLOSING.match(t))


async def intent_node(state: State) -> State:
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.INTENT_RECOGNIZING

    user_text = (current_state.current_question or current_state.user_request or "").strip()
    cfg = getattr(current_state, "config", None) or {}
    chat_history = getattr(current_state, "chat_history", None) or []

    await state_queue.put(
        BackToFrontData(
            step=ExecutionState.INTENT_RECOGNIZING,
            state="processing",
            data="正在识别对话意图...",
        )
    )

    route = "research"
    try:
        # 多轮里已有学术对话时，不用极短规则跳过 LLM
        has_prior = len([m for m in chat_history if m.get("role") == "assistant"]) > 0
        if (
            not has_prior
            and _maybe_short_closing_chat(user_text)
            and len(chat_history) <= 2
        ):
            route = "chat"
            logger.debug("意图：极短结束语，直接判为 chat")
        else:
            prompt = _build_intent_task(user_text, cfg, chat_history)
            result = await _make_intent_agent().run(task=prompt)
            raw_content = result.messages[-1].content
            raw_str = str(raw_content)
            logger.debug("意图识别原始输出: %s", raw_str[:500])
            route = _parse_intent(raw_str)
    except Exception as e:
        logger.warning(f"意图识别失败，默认走 research: {e}")
        route = "research"

    if current_state.config is None:
        current_state.config = {}

    current_state.config["intent_route"] = route

    # 文献意图且未选手动库：创建本会话专属知识库（仅 research，避免闲聊也建库）
    if route == "research":
        c = current_state.config
        if not c.get("bypass_to_qa") and c.get("kb_binding") != "manual":
            if not (c.get("selected_db_ids") or []):
                try:
                    from src.knowledge.session_kb import create_session_research_kb

                    db_id = await create_session_research_kb(user_text)
                    c["selected_db_ids"] = [db_id]
                    c["kb_binding"] = "built"
                except Exception as e:
                    logger.error(f"创建会话知识库失败: {e}")

    await state_queue.put(
        BackToFrontData(
            step=ExecutionState.INTENT_RECOGNIZING,
            state="completed",
            data="闲聊模式" if route == "chat" else "文献检索与学术问答",
        )
    )
    return {"value": current_state}
