import asyncio
import importlib
import sys
import types


class _ConfigStub:
    def __init__(self):
        self._store = {"top_k": 3, "tmp_db_id": "", "current_db_ids": [], "current_db_id": "", "similarity_threshold": 0.0}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def set(self, key, value):
        self._store[key] = value


class _KBStub:
    def __init__(self, query_result, db_info):
        self._query_result = query_result
        self._db_info = db_info

    async def aquery(self, querys, db_id, top_k, similarity_threshold):  # noqa: ARG002
        return self._query_result

    def get_database_info(self, db_id):  # noqa: ARG002
        return self._db_info


def _load_retrieval_tool(query_result, db_info, graph=None):
    kb_module = types.ModuleType("src.knowledge.knowledge")
    kb_module.knowledge_base = _KBStub(query_result, db_info)
    sys.modules["src.knowledge.knowledge"] = kb_module

    cfg_module = types.ModuleType("src.core.config")
    cfg_module.config = _ConfigStub()
    sys.modules["src.core.config"] = cfg_module

    graph_module = types.ModuleType("src.services.graph_store")
    graph_module.load_entity_graph = lambda _db_id: graph
    graph_module.rerank_chunks_by_entity_graph = lambda chunks, _graph, _query, top_k: chunks[:top_k]
    sys.modules["src.services.graph_store"] = graph_module

    import src.services.retrieval_tool as rt

    return importlib.reload(rt)


def test_retrieval_tool_returns_formatted_context_for_selected_db():
    query_result = [{"content": "chunk-a", "metadata": {"source": "s1"}, "score": 0.8}]
    db_info = {"additional_params": {"retrieval_method": "vector"}}
    rt = _load_retrieval_tool(query_result, db_info)
    results = asyncio.run(rt.retrieval_tool(["question"], preferred_db_ids=["db1"], top_k=2, retrieval_mode="rag"))
    assert len(results) == 1 and "知识库：db1" in results[0]


def test_retrieval_tool_both_mode_merges_graph_and_rag_results_without_duplicates():
    query_result = [
        {"content": "chunk-a", "metadata": {"source": "s1"}, "score": 0.8},
        {"content": "chunk-b", "metadata": {"source": "s1"}, "score": 0.7},
    ]
    db_info = {"additional_params": {"retrieval_method": "hybrid"}}
    rt = _load_retrieval_tool(query_result, db_info, graph={"nodes": {}, "edges": {}})
    results = asyncio.run(rt.retrieval_tool(["question"], preferred_db_ids=["db2"], top_k=2, retrieval_mode="both"))
    assert len(results) == 2