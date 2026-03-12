import importlib
import sys
import types


class _ConfigStub:
    def __init__(self, save_dir):
        self._store = {"SAVE_DIR": save_dir, "graphrag": {"hops": 2, "seed_alpha": 0.6}}

    def get(self, key, default=None):
        if "." not in key:
            return self._store.get(key, default)
        value = self._store
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def get_int(self, key, default=0):
        return int(self.get(key, default))

    def get_float(self, key, default=0.0):
        return float(self.get(key, default))

    def __contains__(self, key):
        return self.get(key, None) is not None


def _load_graph_store_module(tmp_path):
    fake_cfg_module = types.ModuleType("src.core.config")
    fake_cfg_module.config = _ConfigStub(str(tmp_path))
    sys.modules["src.core.config"] = fake_cfg_module
    sys.modules.pop("src.services.graph_store", None)
    return importlib.import_module("src.services.graph_store")


def test_rerank_chunks_by_entity_graph_prioritizes_entity_match(tmp_path):
    gs = _load_graph_store_module(tmp_path)
    graph = {
        "db_id": "db1",
        "nodes": {
            "paper:p1": {"id": "paper:p1", "type": "Paper", "label": "P1", "norm_label": "p1"},
            "method:transformer": {
                "id": "method:transformer",
                "type": "Method",
                "label": "Transformer",
                "norm_label": "transformer",
            },
        },
        "edges": [{"source": "paper:p1", "target": "method:transformer", "type": "USES_METHOD", "weight": 1.0}],
        "paper_entities": {"p1": ["paper:p1", "method:transformer"]},
        "entity_aliases": {"transformer": "method:transformer"},
        "stats": {},
    }
    chunks = [
        {"content": "uses transformer blocks", "score": 0.4, "metadata": {"paper_id": "p1"}},
        {"content": "random baseline", "score": 0.9, "metadata": {"paper_id": "p2"}},
    ]
    ranked = gs.rerank_chunks_by_entity_graph(chunks, graph, "transformer", 2)
    assert ranked[0]["metadata"]["paper_id"] == "p1"


def test_rerank_chunks_by_entity_graph_returns_top_k_when_graph_empty(tmp_path):
    gs = _load_graph_store_module(tmp_path)
    chunks = [{"content": "a", "score": 0.2, "metadata": {}}, {"content": "b", "score": 0.1, "metadata": {}}]
    ranked = gs.rerank_chunks_by_entity_graph(chunks, {}, "query", 1)
    assert len(ranked) == 1 and ranked[0]["content"] == "a"