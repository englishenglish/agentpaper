import sys
import types


def _load_retrieval_tool():
    sys.modules.pop("src.services.retrieval_tool", None)

    kb_mod = types.ModuleType("src.knowledge")
    kb_mod.knowledge_base = object()
    sys.modules["src.knowledge"] = kb_mod

    gb_mod = types.ModuleType("src.extraction.graph_builder")
    gb_mod.load_entity_graph = lambda _db_id, embedder=None: None
    sys.modules["src.extraction.graph_builder"] = gb_mod

    import src.services.retrieval_tool as rt
    import src.core.config as cfg_mod

    cfg_mod.config = type("Cfg", (), {"get": lambda self, key, default=None: default})()

    return rt


def test_extract_chunks_from_query_result_list_format():
    rt = _load_retrieval_tool()
    chunks = rt._extract_chunks_from_query_result(
        [{"content": "doc-a", "metadata": {"source": "file-a"}, "score": 0.5}],
        "db1",
    )
    assert len(chunks) == 1 and "知识库: db1" in chunks[0]["formatted"]


def test_extract_chunks_from_query_result_chroma_format():
    rt = _load_retrieval_tool()
    chunks = rt._extract_chunks_from_query_result(
        {"documents": [["doc-a"]], "metadatas": [[{"source": "file-a"}]], "distances": [[0.1]]},
        "db1",
    )
    assert len(chunks) == 1 and chunks[0]["score"] == 0.9


def test_rag_rerank_prefers_keyword_overlap():
    rt = _load_retrieval_tool()
    ranked = rt._rag_rerank(
        [
            {"content": "contains transformer", "score": 0.4},
            {"content": "unrelated", "score": 0.4},
        ],
        "transformer",
        2,
    )
    assert ranked[0]["content"] == "contains transformer"
