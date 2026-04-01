from __future__ import annotations

import asyncio
import os
import traceback
from typing import Any, List, Optional, Union

import chromadb
from chromadb.errors import NotFoundError
import httpx
import numpy as np
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings,
    Space,
)
from chromadb.config import Settings
from openai import OpenAI

from src.core.config import config
from src.knowledge.base import KnowledgeBase
from src.knowledge.indexing import process_file_to_markdown, process_url_to_markdown
from src.knowledge.utils.embedding_sentence_chunk import embedding_sentence_chunk_chunks
from src.knowledge.utils.kb_utils import (
    get_embedding_config,
    prepare_item_metadata,
    validate_img_embedding_file,
)
from src.utils.datetime_utils import utc_isoformat
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------------------------------
# Reranker（懒加载单例，仅在 enable_reranker=true 时使用）
# ------------------------------------------------------------------

_GLOBAL_RERANKER = None


def _get_reranker():
    """懒加载 BGEReranker，首次调用时初始化。"""
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        logger.info("正在加载 BGEReranker 模型到内存中 (首次查询会稍慢)...")
        from src.knowledge.rerank import BGEReranker
        _GLOBAL_RERANKER = BGEReranker()
    return _GLOBAL_RERANKER


# ------------------------------------------------------------------
# Embedding 函数
# ------------------------------------------------------------------

class ResilientOpenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    基于自定义 httpx.Client 的 OpenAI 兼容 Embedding 函数。

    - `trust_env` 默认为 False，避免系统代理导致 TLS 握手超时。
    - 超时时间可通过构造参数配置。
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: str,
        dimensions: Optional[int] = None,
        trust_env: bool = False,
        timeout_seconds: float = 120.0,
        connect_seconds: float = 45.0,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        base = (api_base or "").replace("/embeddings", "").rstrip("/")
        timeout = httpx.Timeout(timeout_seconds, connect=connect_seconds)
        http_client = httpx.Client(timeout=timeout, trust_env=trust_env)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base,
            http_client=http_client,
            timeout=timeout,
            max_retries=2,
        )
        logger.info(
            "Embedding HTTP client: base_url=%s trust_env=%s timeout=%ss connect=%ss",
            base,
            trust_env,
            timeout_seconds,
            connect_seconds,
        )

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        params: dict[str, Any] = {"model": self.model_name, "input": input}
        if self.dimensions is not None and "text-embedding-3" in self.model_name:
            params["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**params)
        return [np.array(data.embedding, dtype=np.float32) for data in response.data]

    @staticmethod
    def name() -> str:
        # 须与 Chroma 持久化集合中的 embedding 名称一致（历史数据多为 "openai"），
        # 否则 get_collection 会报 embedding function conflict，且不可误走 create。
        return "openai"

    def default_space(self) -> Space:
        return "cosine"


# ------------------------------------------------------------------
# ChromaKB
# ------------------------------------------------------------------

class ChromaKB(KnowledgeBase):
    """基于 ChromaDB 的向量知识库实现。"""

    def __init__(self, work_dir: str, **kwargs) -> None:
        super().__init__(work_dir)

        self.chroma_db_path = os.path.join(work_dir, "chromadb")
        os.makedirs(self.chroma_db_path, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # {db_id: collection}（含图片集合 {db_id}_images）
        self.collections: dict[str, Any] = {}
        logger.info("ChromaKB initialized")

    @property
    def kb_type(self) -> str:
        return "chroma"

    # ------------------------------------------------------------------
    # 内部：集合管理
    # ------------------------------------------------------------------

    async def _create_kb_instance(self, db_id: str, _config: dict) -> Any:
        """创建或获取 ChromaDB 向量集合。"""
        if db_id not in self.databases_meta:
            raise ValueError(f"Database {db_id} not found")

        embed_info = self.databases_meta[db_id].get("embed_info") or {}
        embedding_function = self._get_embedding_function(embed_info)
        collection_name = db_id

        try:
            collection = self.chroma_client.get_collection(
                name=collection_name, embedding_function=embedding_function
            )
            logger.info(f"Retrieved existing collection: {collection_name}")
        except NotFoundError:
            logger.info(
                "Creating new collection with embedding model: %s",
                embed_info.get("name", "default"),
            )
            collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={
                    "db_id": db_id,
                    "created_at": utc_isoformat(),
                    "embedding_model": embed_info.get("name", "default"),
                    "hnsw:space": "cosine",
                },
            )
            logger.info(f"Created new collection: {collection_name}")

        return collection

    async def _initialize_kb_instance(self, instance: Any) -> None:
        """无需特殊初始化。"""

    def _get_embedding_function(self, embed_info: dict) -> ResilientOpenAIEmbeddingFunction:
        """根据 embed_info 构造 Embedding 函数实例。"""
        cfg = get_embedding_config(embed_info)
        trust_env = config.get_bool("embedding_http_trust_env", False)
        timeout_s = float(config.get("embedding_http_timeout", 120) or 120)
        connect_s = float(config.get("embedding_http_connect", 45) or 45)
        dim = cfg.get("dimension")
        return ResilientOpenAIEmbeddingFunction(
            model_name=cfg["model"],
            api_key=cfg["api_key"],
            api_base=cfg["base_url"],
            dimensions=dim if isinstance(dim, int) else None,
            trust_env=trust_env,
            timeout_seconds=timeout_s,
            connect_seconds=connect_s,
        )

    async def _get_chroma_collection(self, db_id: str) -> Any | None:
        """获取或懒创建文本向量集合；失败时返回 None。"""
        if db_id in self.collections:
            return self.collections[db_id]

        if db_id not in self.databases_meta:
            logger.warning(f"Database {db_id} not found in metadata")
            return None

        try:
            collection = await self._create_kb_instance(db_id, {})
            await self._initialize_kb_instance(collection)
            self.collections[db_id] = collection
            return collection
        except Exception as e:
            logger.error(f"Failed to create vector collection for {db_id}: {e}\n{traceback.format_exc()}")
            return None

    async def _get_image_chroma_collection(self, db_id: str) -> Any | None:
        """获取或懒创建图片专用集合（512 维，不使用外部 Embedding 函数）。"""
        if db_id not in self.databases_meta:
            return None

        image_collection_name = f"{db_id}_images"
        if image_collection_name in self.collections:
            return self.collections[image_collection_name]

        try:
            try:
                collection = self.chroma_client.get_collection(name=image_collection_name)
                logger.info(f"Retrieved existing image collection: {image_collection_name}")
            except Exception:
                class _ZeroEmbedFn:
                    """占位 Embedding 函数——实际嵌入由外部 CLIP 模型生成。"""
                    def __call__(self, texts):
                        return [[0.0] * 512 for _ in texts]

                collection = self.chroma_client.create_collection(
                    name=image_collection_name,
                    embedding_function=_ZeroEmbedFn(),
                    metadata={
                        "db_id": db_id,
                        "created_at": utc_isoformat(),
                        "embedding_model": "clip_image_embedding",
                        "embedding_dimension": 512,
                        "hnsw:space": "cosine",
                    },
                )
                logger.info(f"Created new image collection: {image_collection_name}")

            self.collections[image_collection_name] = collection
            return collection
        except Exception as e:
            logger.error(f"Failed to get/create image collection {image_collection_name}: {e}")
            return None

    # ------------------------------------------------------------------
    # 写入接口
    # ------------------------------------------------------------------

    async def add_processed_content(self, db_id: str, data: dict | None = None) -> list[dict]:
        """将已切好的 chunks 批量写入 ChromaDB（每批 10 条）。"""
        if db_id not in self.databases_meta:
            raise ValueError(f"Database {db_id} not found")

        collection = await self._get_chroma_collection(db_id)
        if not collection:
            raise ValueError(f"Failed to get ChromaDB collection for {db_id}")

        try:
            batch_documents = (data or {}).get("documents", [])
            if not batch_documents:
                return []

            batch_embeddings = data.get("embeddings")
            batch_metadatas = data.get("metadatas")
            batch_ids = data.get("ids")
            batch_size = 10

            for i in range(0, len(batch_documents), batch_size):
                sl = slice(i, i + batch_size)
                await asyncio.to_thread(
                    collection.add,
                    embeddings=batch_embeddings[sl] if batch_embeddings else None,
                    documents=batch_documents[sl],
                    metadatas=batch_metadatas[sl] if batch_metadatas else None,
                    ids=batch_ids[sl] if batch_ids else None,
                )

            logger.info(f"Inserted {len(batch_documents)} chunks into database {db_id}")
        except Exception as e:
            logger.error(f"Failed to insert {len(batch_documents)} chunks into {db_id}: {e}\n{traceback.format_exc()}")

        return []

    async def add_content(self, db_id: str, items: list[str], params: dict | None = None) -> list[dict]:
        """处理文件/URL 并写入 ChromaDB。"""
        if db_id not in self.databases_meta:
            raise ValueError(f"Database {db_id} not found")

        collection = await self._get_chroma_collection(db_id)
        if not collection:
            raise ValueError(f"Failed to get ChromaDB collection for {db_id}")

        content_type = (params or {}).get("content_type", "file")
        processed_items_info = []

        for item in items:
            metadata = prepare_item_metadata(item, content_type, db_id)
            file_id = metadata["file_id"]
            filename = metadata["filename"]

            file_record = metadata.copy()
            self.files_meta[file_id] = file_record
            self._save_metadata()
            self._add_to_processing_queue(file_id)

            try:
                if content_type == "file":
                    markdown_content = await process_file_to_markdown(item, params=params)
                else:
                    markdown_content = await process_url_to_markdown(item, params=params)

                p = params or {}
                chunks = embedding_sentence_chunk_chunks(markdown_content, file_id, filename, p)
                logger.info(f"Split {filename} into {len(chunks)} chunks (embedding_sentence_chunk_chunks)")

                if chunks:
                    batch_size = 10  # DashScope 等供应商的安全上限
                    total = len(chunks)
                    for i in range(0, total, batch_size):
                        sl = slice(i, i + batch_size)
                        await asyncio.to_thread(
                            collection.add,
                            documents=[c["content"] for c in chunks[sl]],
                            metadatas=[c["metadata"] for c in chunks[sl]],
                            ids=[c["id"] for c in chunks[sl]],
                        )
                        logger.info(f"Batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} for {filename}")

                self.files_meta[file_id]["status"] = "done"
                file_record["status"] = "done"
            except Exception as e:
                logger.error(f"Failed to process {content_type} {item}: {e}\n{traceback.format_exc()}")
                self.files_meta[file_id]["status"] = "failed"
                file_record["status"] = "failed"
            finally:
                self._save_metadata()
                self._remove_from_processing_queue(file_id)

            processed_items_info.append(file_record)

        return processed_items_info

    async def add_image_embeddings(self, db_id: str, item: str, params: dict | None = None) -> list[dict]:
        """处理图片嵌入文件（JSON）并写入图片专用集合。"""
        if not validate_img_embedding_file(item):
            return []
        if db_id not in self.databases_meta:
            raise ValueError(f"Database {db_id} not found")

        collection = await self._get_image_chroma_collection(db_id)
        if not collection:
            raise ValueError(f"Failed to get image ChromaDB collection for {db_id}")

        content_type = (params or {}).get("content_type", "file")
        metadata = prepare_item_metadata(item, content_type, db_id)
        file_id = metadata["file_id"]
        filename = metadata["filename"]

        file_record = metadata.copy()
        self.files_meta[file_id] = file_record
        self._save_metadata()
        self._add_to_processing_queue(file_id)

        try:
            from src.knowledge.indexing import process_file_to_json
            await process_file_to_json(item, params=params)

            # parse_json_into_embedding_chunks 由子类或外部实现；此处作基类占位
            chunks: list[dict] = []
            logger.info(f"Split {filename} into {len(chunks)} image embedding chunks")

            if chunks:
                batch_size = 64
                total = len(chunks)
                for i in range(0, total, batch_size):
                    sl = slice(i, i + batch_size)
                    await asyncio.to_thread(
                        collection.add,
                        documents=[c["content"] for c in chunks[sl]],
                        embeddings=[c["embeddings"] for c in chunks[sl]],
                        metadatas=[c["metadata"] for c in chunks[sl]],
                        ids=[c["id"] for c in chunks[sl]],
                    )
                    logger.info(f"Batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} for {filename}")

            self.files_meta[file_id]["status"] = "done"
            file_record["status"] = "done"
            logger.info(f"Inserted image embeddings for {item} into {db_id}")
        except Exception as e:
            logger.error(f"Failed to process image embeddings for {item}: {e}\n{traceback.format_exc()}")
            self.files_meta[file_id]["status"] = "failed"
            file_record["status"] = "failed"
            raise
        finally:
            self._save_metadata()
            self._remove_from_processing_queue(file_id)

        return [file_record]

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    async def aquery(self, db_id: str, query_text: Union[str, List[str]] = "", **kwargs) -> list[dict]:
        """
        异步查询 ChromaDB 集合。

        当 `enable_reranker=true` 时，先大倍率召回再用 BGEReranker 精排；
        否则直接按向量相似度返回 top_k 结果。
        """
        if not query_text:
            raise ValueError("query_text cannot be empty")

        collection = await self._get_chroma_collection(db_id)
        if not collection:
            raise ValueError(f"Database {db_id} not found")

        query_texts = [query_text] if isinstance(query_text, str) else list(query_text)
        final_top_k: int = kwargs.get("top_k", config.get("top_k", 5))
        similarity_threshold: float = float(kwargs.get("similarity_threshold", 0.0))
        enable_reranker: bool = config.get_bool("enable_reranker", False)

        try:
            if enable_reranker:
                return await self._aquery_with_reranker(
                    collection, query_texts, final_top_k
                )
            else:
                return await self._aquery_vector_only(
                    collection, query_texts, final_top_k, similarity_threshold
                )
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}\n{traceback.format_exc()}")
            return []

    async def _aquery_vector_only(
        self,
        collection: Any,
        query_texts: list[str],
        top_k: int,
        similarity_threshold: float,
    ) -> list[dict]:
        """仅向量相似度召回，按分数过滤后返回 top_k。"""
        raw = await asyncio.to_thread(
            collection.query,
            query_texts=query_texts,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = raw["documents"][0] if raw.get("documents") else []
        metadatas = raw["metadatas"][0] if raw.get("metadatas") else []
        distances = raw["distances"][0] if raw.get("distances") else []

        results: list[dict] = []
        seen: set[str] = set()
        for i, doc in enumerate(documents):
            if not doc:
                continue
            meta = (metadatas[i] if i < len(metadatas) else {}) or {}
            if "full_doc_id" in meta:
                meta["file_id"] = meta.pop("full_doc_id")
            distance = distances[i] if i < len(distances) else 1.0
            score = float(1.0 - distance)
            if score < similarity_threshold:
                continue
            chunk_id = meta.get("chunk_id")
            if chunk_id and chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)
            results.append({"content": doc, "metadata": meta, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def _aquery_with_reranker(
        self,
        collection: Any,
        query_texts: list[str],
        final_top_k: int,
    ) -> list[dict]:
        """先大倍率向量召回，再用 BGEReranker 精排后返回 final_top_k。"""
        recall_k = final_top_k * 4

        raw = await asyncio.to_thread(
            collection.query,
            query_texts=query_texts,
            n_results=recall_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = raw["documents"][0] if raw.get("documents") else []
        metadatas = raw["metadatas"][0] if raw.get("metadatas") else []

        if not documents:
            return []

        reranker = _get_reranker()
        scores: list[float] = await asyncio.to_thread(
            reranker.rerank, query_texts[0], documents
        )

        ranked: list[dict] = []
        for i, doc in enumerate(documents):
            meta = (metadatas[i] if i < len(metadatas) else {}) or {}
            if "full_doc_id" in meta:
                meta["file_id"] = meta.pop("full_doc_id")
            ranked.append({
                "content": doc,
                "metadata": meta,
                "score": float(scores[i]),
                "original_chunk_id": meta.get("chunk_id"),
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)

        results: list[dict] = []
        seen: set[str] = set()
        for chunk in ranked:
            chunk_id = chunk.get("original_chunk_id")
            if chunk_id and chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)
            results.append(chunk)
            if len(results) >= final_top_k:
                break

        logger.info(f"Reranked {len(documents)} docs → top {len(results)}")
        return results

    # ------------------------------------------------------------------
    # 文件管理
    # ------------------------------------------------------------------

    async def delete_file(self, db_id: str, file_id: str) -> None:
        """从 ChromaDB 集合及 files_meta 中删除文件及其所有 chunks。"""
        collection = await self._get_chroma_collection(db_id)
        if collection:
            try:
                results = collection.get(where={"full_doc_id": file_id}, include=["metadatas"])
                if results and results.get("ids"):
                    collection.delete(ids=results["ids"])
                    logger.info(f"Deleted {len(results['ids'])} chunks for file {file_id}")
            except Exception as e:
                logger.error(f"Error deleting file {file_id} from ChromaDB: {e}")

        if file_id in self.files_meta:
            del self.files_meta[file_id]
            self._save_metadata()

    async def get_file_basic_info(self, db_id: str, file_id: str) -> dict:
        """获取文件基本元数据。"""
        if file_id not in self.files_meta:
            raise FileNotFoundError(f"File not found: {file_id}")
        return {"meta": self.files_meta[file_id]}

    async def get_file_content(self, db_id: str, file_id: str) -> dict:
        """获取文件 chunks（按 chunk_order_index 排序）。"""
        if file_id not in self.files_meta:
            raise FileNotFoundError(f"File not found: {file_id}")

        content_info: dict = {"lines": []}
        collection = await self._get_chroma_collection(db_id)
        if not collection:
            return content_info

        try:
            results = collection.get(
                where={"full_doc_id": file_id},
                include=["documents", "metadatas"],
            )
            doc_chunks: list[dict] = []
            if results and results.get("ids"):
                for i, chunk_id in enumerate(results["ids"]):
                    meta_i = results["metadatas"][i] if i < len(results["metadatas"]) else {}
                    doc_chunks.append({
                        "id": chunk_id,
                        "content": results["documents"][i] if i < len(results["documents"]) else "",
                        "metadata": meta_i,
                        "chunk_order_index": meta_i.get("chunk_index", i),
                    })

            doc_chunks.sort(key=lambda x: x.get("chunk_order_index", 0))
            content_info["lines"] = doc_chunks
        except Exception as e:
            logger.error(f"Failed to get file content from ChromaDB: {e}")

        return content_info

    async def get_file_info(self, db_id: str, file_id: str) -> dict:
        """获取文件完整信息（基本信息 + chunks）- 保持向后兼容。"""
        if file_id not in self.files_meta:
            raise FileNotFoundError(f"File not found: {file_id}")
        basic = await self.get_file_basic_info(db_id, file_id)
        content = await self.get_file_content(db_id, file_id)
        return {**basic, **content}
