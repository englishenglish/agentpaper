import asyncio
import sys
import types


class _ConfigStub:
    def __init__(self):
        self._store = {
            "top_k": 3,
            "tmp_db_id": "",
            "current_db_ids": [],
            "current_db_id": "",
            "similarity_threshold": 0.0,
        }

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
    """Stub heavy deps so retrieval_tool loads without pulling Chroma/httpx."""
    sys.modules.pop("src.services.retrieval_tool", None)

    kb_mod = types.ModuleType("src.knowledge")
    kb_mod.knowledge_base = _KBStub(query_result, db_info)
    sys.modules["src.knowledge"] = kb_mod

    gb_mod = types.ModuleType("src.extraction.graph_builder")
    gb_mod.load_entity_graph = lambda _db_id, embedder=None: graph
    sys.modules["src.extraction.graph_builder"] = gb_mod

    import src.services.retrieval_tool as rt
    import src.core.config as cfg_mod

    cfg_mod.config = _ConfigStub()

    return rt


def test_retrieval_tool_returns_formatted_context_for_selected_db():
    query_result = [{"content": "chunk-a", "metadata": {"source": "s1"}, "score": 0.8}]
    db_info = {"additional_params": {"retrieval_method": "rag"}}
    rt = _load_retrieval_tool(query_result, db_info)
    formatted, cites = asyncio.run(
        rt.retrieval_tool(["question"], preferred_db_ids=["db1"], top_k=2, retrieval_mode="rag")
    )
    assert len(formatted) == 1 and "知识库: db1" in formatted[0]
    assert len(cites) == 1 and cites[0].get("ref") == 1


def test_retrieval_tool_both_mode_merges_graph_and_rag_results_without_duplicates():
    query_result = [
        {"content": "chunk-a", "metadata": {"source": "s1"}, "score": 0.8},
        {"content": "chunk-b", "metadata": {"source": "s1"}, "score": 0.7},
    ]
    db_info = {"additional_params": {"retrieval_method": "rag"}}
    # 无实体图时 both 走轻量 _graphrag_rerank，避免依赖真实嵌入与图谱结构
    rt = _load_retrieval_tool(query_result, db_info, graph=None)
    formatted, cites = asyncio.run(
        rt.retrieval_tool(["question"], preferred_db_ids=["db2"], top_k=2, retrieval_mode="both")
    )
    assert len(formatted) == 2
    assert len(cites) == 2
