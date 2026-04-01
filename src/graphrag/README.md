# `src/graphrag` 与 `src/rag` 说明

本文档介绍 **`src/graphrag`**（论文实体图谱构建与可选 Neo4j），并**详细说明 `src/rag`**（向量知识库 / 经典 RAG）目录下各文件职责。二者常与 **`src/retriever`** 中的 `GraphRAGRetriever`、`HybridRetriever` 一起使用：前者偏「图结构与社区」，后者偏「向量召回与混合编排」。

---

## 和「RAG」的关系（一句话）

| 概念 | 目录 / 入口 | 做什么 |
|------|-------------|--------|
| **向量 RAG** | `src/rag/` | 文档解析、切块、写入 Chroma、查询、HTTP 管理接口 |
| **GraphRAG（图谱侧）** | `src/graphrag/` | 从论文结构化结果建实体图、持久化、社区摘要、可选 Neo4j |
| **检索编排** | `src/retriever/` | `RagRetriever`、`GraphRAGRetriever`、`HybridRetriever`（rag / graphrag / both） |

图谱检索实现类不在 `src/graphrag` 内，而在 **`src/retriever/graphrag_retriever.py`**（也可 `from src.retriever import GraphRAGRetriever`）。

---

## `src/graphrag/` 文件

| 文件 | 说明 |
|------|------|
| `schema.py` | 实体/关系类型、标签归一化、缩写等**共享常量与工具**，供建图与检索对齐概念。 |
| `graph_builder.py` | 从结构化输出**构建实体图**、持久化为 JSON、**LRU 缓存**加载，供可视化或检索使用。 |
| `community_builder.py` | 基于图的 **Louvain 社区划分**与 **LLM 社区摘要**（用于粗粒度主题归纳）。 |
| `neo4j_client.py` | **可选** Neo4j 客户端与全量同步；未部署 Neo4j 时仍可仅用本地 JSON 图谱。 |
| `__init__.py` | 包说明与推荐导入方式（按需导入子模块，检索类从 `src.retriever` 取）。 |

---

## `src/rag/` 文件（向量 RAG 详解）

整体数据流：**抽象接口 → 工厂选实现 → 管理器统一切 `db_id` → Chroma 存向量 → HTTP 暴露增删查**。

### 核心架构

| 文件 | 说明 |
|------|------|
| `base.py` | 抽象基类 **`KnowledgeBase`**：元数据（库/文件）、创建库、增删文档、`aquery` 等统一接口；异常类型 **`KBNotFoundError`** 等。所有具体后端（如 Chroma）都继承它。 |
| `factory.py` | **`KnowledgeBaseFactory`**：用字符串 **`kb_type`**（如 `"chroma"`）**注册**实现类，并 **`create()`** 创建实例，避免在业务里写死 `if chroma: ...`。 |
| `manager.py` | **`KnowledgeBaseManager`**：**全局入口**：维护 `global_metadata.json`、按 `db_id` 路由到对应 `KnowledgeBase` 实例；提供 `create_database`、`add_content`、`aquery`、`list_database_documents` 等，Agent 与 API 主要和它打交道。 |
| `router.py` | FastAPI **`APIRouter`**（前缀 `/knowledge`）：知识库 CRUD、文档上传/列表/删除、查询测试、图谱调试接口 `GET .../graph`、上传校验等。依赖全局 **`knowledge_base`** 与 **`config`**。 |
| `session_kb.py` | **`create_session_research_kb`**：在未选手动知识库时，为联网检索**自动创建会话级知识库**（名称、嵌入配置来自 `config`）。 |

### 实现与索引

| 文件 | 说明 |
|------|------|
| `implementations/chroma.py` | **`ChromaKB`**：基于 **ChromaDB** 的具体实现——集合管理、OpenAI 兼容 Embedding、文件/Markdown 切块写入、`aquery`（可选 **BGE 重排**）、图片集合占位逻辑等。 |
| `indexing.py` | **入库前解析**：支持扩展名列表、`process_file_to_markdown` / `process_url_to_markdown`、PDF/Word/纯文本等读取与 OCR 相关分支（与具体 OCR 模块配合）。把「文件 → 文本」阶段集中在这里。 |
| `rerank.py` | **`BGEReranker`**：基于 **FlagEmbedding** 的交叉编码器重排；在配置开启时由 Chroma 查询路径**懒加载**调用。 |

### 工具与子模块 `utils/`

| 文件 | 说明 |
|------|------|
| `utils/kb_utils.py` | 上传路径校验（防路径穿越）、**内容哈希**、`prepare_item_metadata`、**从 config 解析 embedding 连接信息** `get_embedding_config` 等。 |
| `utils/embedding_sentence_chunk.py` | **按句向量相似度切块**（相邻句 cosine 决策是否合并），与「按标题切块」不同，依赖 **共享 Embedder**（`src.core.embedding`）。 |
| `utils/__init__.py` | 对常用函数再导出，供 `from src.rag...utils import ...` 简短导入。 |

### 包入口

| 文件 | 说明 |
|------|------|
| `__init__.py` | 对上层导出 **`knowledge_base`** 等（见文件内 `__all__`）。 |

---

## 检索与配置（交叉引用）

- **向量检索封装**：`src/retriever/rag_retriever.py`（通常通过 `knowledge_base.aquery` 访问底层库）。
- **混合模式**：`src/retriever/hybrid_retriever.py`（`rag` / `graphrag` / `both`）。
- **嵌入与模型**：`src/core/embedding.py`、`src/core/models.yaml`、`src/core/system_params.yaml`（如 `SAVE_DIR`、是否启用 reranker）。

若只关心「图文件长什么样」，看 **`src/graphrag/graph_builder.py` 与持久化目录**；若只关心「文档怎么进向量库」，跟 **`src/rag/indexing.py` → `implementations/chroma.py`** 这条链即可。
