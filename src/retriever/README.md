# `src/retriever` — 检索实现

| 文件 | 说明 |
|------|------|
| `rag_retriever.py` | 仅含 ``RagRetriever``：分词/格式化/解析/重排/引用等均为类方法。 |
| `graphrag_retriever.py` | `GraphRAGRetriever`：实体图谱五分量重排（依赖 `src.graphrag.schema`）。 |
| `hybrid_retriever.py` | `HybridRetriever`：编排上述二者；`retrieval_tool`、`normalize_retrieval_mode`。 |

包入口 ``from src.retriever import GraphRAGRetriever, HybridRetriever, RagRetriever, retrieval_tool``。
