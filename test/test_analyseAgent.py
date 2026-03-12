import importlib
import json
import sys
import types


class _ConfigStub:
    def __init__(self):
        self._store = {"SAVE_DIR": "."}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def set(self, key, value):
        self._store[key] = value

    def get_int(self, key, default=0):  # noqa: ARG002
        return default

    def get_float(self, key, default=0.0):  # noqa: ARG002
        return default

    def __contains__(self, key):
        return key in self._store


def _load_graph_store_module(tmp_path):
    fake_cfg_module = types.ModuleType("src.core.config")
    cfg = _ConfigStub()
    cfg.set("SAVE_DIR", str(tmp_path))
    fake_cfg_module.config = cfg
    sys.modules["src.core.config"] = fake_cfg_module
    import src.services.graph_store as graph_store

    return importlib.reload(graph_store)


def test_build_entity_graph_contains_expected_types(tmp_path):
    gs = _load_graph_store_module(tmp_path)
    papers = [{"paper_id": "p1", "title": "GraphRAG Paper"}]
    extracted = [
        {
            "key_methodology": {"name": "Method A"},
            "datasets_used": ["Dataset A"],
            "evaluation_metrics": ["F1"],
            "core_problem": "Retrieval quality",
            "contributions": ["Better rerank"],
        }
    ]
    graph = gs.build_entity_graph_from_papers(papers, extracted, "db1")
    type_count = graph["stats"]["node_type_count"]
    assert type_count["Paper"] == 1 and type_count["Method"] == 1 and type_count["Dataset"] == 1


def test_save_and_load_entity_graph_roundtrip(tmp_path):
    gs = _load_graph_store_module(tmp_path)
    graph = {"db_id": "dbx", "nodes": {}, "edges": [], "paper_entities": {}, "entity_aliases": {}, "stats": {}}
    path = gs.save_entity_graph("dbx", graph)
    loaded = gs.load_entity_graph("dbx")
    assert path.endswith("dbx.json") and loaded == graph


def test_load_entity_graph_returns_none_for_missing_file(tmp_path):
    gs = _load_graph_store_module(tmp_path)
    loaded = gs.load_entity_graph("missing")
    assert loaded is None


def test_graph_summary_returns_default_for_empty_graph(tmp_path):
    gs = _load_graph_store_module(tmp_path)
    summary = gs.graph_summary({})
    assert summary == {"node_count": 0, "edge_count": 0, "paper_count": 0, "node_type_count": {}}
   