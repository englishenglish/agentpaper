from src.utils.tool_utils import handlerChunk


def test_handler_chunk_switches_to_thinking_state():
    state, is_thinking = handlerChunk(False, "<think>正在分析")
    assert state == "thinking" and is_thinking is True


def test_handler_chunk_ignores_think_start_delimiter_only():
    state, is_thinking = handlerChunk(False, "<think>")
    assert state is None and is_thinking is True


def test_handler_chunk_switches_back_to_generating_on_end_tag():
    state, is_thinking = handlerChunk(True, "</think>结论")
    assert state == "generating" and is_thinking is False


def test_handler_chunk_keeps_thinking_for_normal_chunk_when_flag_true():
    state, is_thinking = handlerChunk(True, "继续思考中")
    assert state == "thinking" and is_thinking is True
