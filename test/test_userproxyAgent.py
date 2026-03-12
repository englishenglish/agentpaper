import importlib
import sys
import types


def _load_retrieval_tool():
    kb_module = types.ModuleType("src.knowledge.knowledge")
    kb_module.knowledge_base = object()
    sys.modules["src.knowledge.knowledge"] = kb_module

    cfg_module = types.ModuleType("src.core.config")
    cfg_module.config = type("Cfg", (), {"get": lambda self, key, default=None: default})()
    sys.modules["src.core.config"] = cfg_module

    graph_module = types.ModuleType("src.services.graph_store")
    graph_module.load_entity_graph = lambda _db_id: None
    graph_module.rerank_chunks_by_entity_graph = lambda chunks, _graph, _query, top_k: chunks[:top_k]
    sys.modules["src.services.graph_store"] = graph_module

    import src.services.retrieval_tool as rt

    return importlib.reload(rt)


def test_tokenize_supports_chinese_and_english():
    rt = _load_retrieval_tool()
    tokens = rt._tokenize("GraphRAG 检索增强")
    assert "graphrag" in tokens and "检索增强" in tokens


def test_rag_rerank_limits_same_source_to_improve_diversity():
    rt = _load_retrieval_tool()
    chunks = [
        {"content": "q a", "score": 0.9, "source": "s1"},
        {"content": "q b", "score": 0.8, "source": "s1"},
        {"content": "q c", "score": 0.7, "source": "s1"},
        {"content": "q d", "score": 0.6, "source": "s2"},
    ]
    ranked = rt._rag_rerank(chunks, "q", 4)
    assert len([item for item in ranked if item["source"] == "s1"]) <= 2


def test_merge_ranked_chunks_deduplicates_same_source_and_content():
    rt = _load_retrieval_tool()
    merged = rt._merge_ranked_chunks(
        [{"content": "same", "source": "s1"}, {"content": "x", "source": "s2"}],
        [{"content": "same", "source": "s1"}, {"content": "y", "source": "s3"}],
        3,
    )
    assert len(merged) == 3