"""
GraphRAG 子系统：论文图谱构建、社区摘要、Neo4j 客户端。

图谱检索类 ``GraphRAGRetriever`` 位于 ``src.retriever.graphrag_retriever``，
也可 ``from src.retriever import GraphRAGRetriever``。

推荐按需导入，例如::

    from src.graphrag.graph_builder import load_entity_graph, GraphBuilder
    from src.retriever import GraphRAGRetriever
    from src.graphrag.schema import VALID_ENTITY_TYPES
"""
