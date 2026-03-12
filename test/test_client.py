import json


def parse_sse_line(line: bytes):
    decoded_line = line.decode("utf-8").strip()
    if not decoded_line.startswith("data: "):
        return None
    try:
        return json.loads(decoded_line[6:])
    except json.JSONDecodeError:
        return None


def test_parse_sse_line_parses_valid_payload():
    parsed = parse_sse_line(b'data: {"step":"qa","state":"completed"}')
    assert parsed == {"step": "qa", "state": "completed"}


def test_parse_sse_line_returns_none_for_non_data_prefix():
    parsed = parse_sse_line(b"event: message")
    assert parsed is None


def test_parse_sse_line_returns_none_for_invalid_json():
    parsed = parse_sse_line(b"data: {invalid")
    assert parsed is None