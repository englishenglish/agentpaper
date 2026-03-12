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


def test_extract_chunks_from_query_result_list_format():
    rt = _load_retrieval_tool()
    chunks = rt._extract_chunks_from_query_result(
        [{"content": "doc-a", "metadata": {"source": "file-a"}, "score": 0.5}],
        "db1",
    )
    assert len(chunks) == 1 and "知识库：db1" in chunks[0]["formatted"]


def test_extract_chunks_from_query_result_chroma_format():
    rt = _load_retrieval_tool()
    chunks = rt._extract_chunks_from_query_result(
        {"documents": [["doc-a"]], "metadatas": [[{"source": "file-a"}]], "distances": [[0.1]]},
        "db1",
    )
    assert len(chunks) == 1 and chunks[0]["score"] == 0.9


def test_hybrid_rerank_prefers_keyword_overlap():
    rt = _load_retrieval_tool()
    ranked = rt._hybrid_rerank(
        [
            {"content": "contains transformer", "score": 0.4},
            {"content": "unrelated", "score": 0.4},
        ],
        "transformer",
        2,
    )
    assert ranked[0]["content"] == "contains transformer"
