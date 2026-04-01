"""向后兼容：从 ``src.retriever`` 再导出混合检索 API。"""

from src.retriever import (
    MODE_BOTH,
    MODE_GRAPHRAG,
    MODE_RAG,
    GraphRAGRetriever,
    HybridRetriever,
    RagRetriever,
    normalize_retrieval_mode,
    retrieval_tool,
)

__all__ = [
    "MODE_BOTH",
    "MODE_GRAPHRAG",
    "MODE_RAG",
    "GraphRAGRetriever",
    "HybridRetriever",
    "RagRetriever",
    "normalize_retrieval_mode",
    "retrieval_tool",
]
