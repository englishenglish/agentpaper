from src.core.config import Config


def test_config_set_and_get_nested_value():
    cfg = Config()
    cfg.set("unit.section.value", "ok")
    assert cfg.get("unit.section.value") == "ok"


def test_config_get_bool_for_truthy_string():
    cfg = Config()
    cfg.set("unit.bool", "true")
    assert cfg.get_bool("unit.bool") is True


def test_config_get_int_with_invalid_value_returns_default():
    cfg = Config()
    cfg.set("unit.int", "not-an-int")
    assert cfg.get_int("unit.int", 9) == 9


def test_config_get_list_from_comma_string():
    cfg = Config()
    cfg.set("unit.list", "a,b, c")
    assert cfg.get_list("unit.list") == ["a", "b", "c"]