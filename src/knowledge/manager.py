from __future__ import annotations

import asyncio
import json
import os
from typing import List, Union

from src.knowledge.base import KBNotFoundError, KnowledgeBase
from src.knowledge.factory import KnowledgeBaseFactory
from src.utils.datetime_utils import coerce_any_to_utc_datetime, utc_isoformat
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

# 全局默认知识库类型（通过 __init__.py 注册）
_DEFAULT_KB_TYPE = "chroma"


class KnowledgeBaseManager:
    """
    知识库管理器。

    统一管理多种类型的知识库实例，为上层业务提供与具体实现无关的统一接口。
    """

    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        # {kb_type: KnowledgeBase 实例}
        self.kb_instances: dict[str, KnowledgeBase] = {}

        # {db_id: 含 kb_type 的全局元数据}
        self.global_databases_meta: dict[str, dict] = {}

        self._metadata_lock = asyncio.Lock()

        self._load_global_metadata()
        self._normalize_global_metadata()
        self._initialize_existing_kbs()

        logger.info("KnowledgeBaseManager initialized")

    # ------------------------------------------------------------------
    # 元数据持久化
    # ------------------------------------------------------------------

    def _load_global_metadata(self) -> None:
        """从磁盘加载全局元数据。"""
        meta_file = os.path.join(self.work_dir, "global_metadata.json")
        if not os.path.exists(meta_file):
            return
        try:
            with open(meta_file, encoding="utf-8") as f:
                data = json.load(f)
            self.global_databases_meta = data.get("databases", {})
            logger.info(f"Loaded global metadata for {len(self.global_databases_meta)} databases")
        except Exception as e:
            logger.error(f"Failed to load global metadata: {e}")

    def _save_global_metadata(self) -> None:
        """将全局元数据持久化到磁盘。"""
        meta_file = os.path.join(self.work_dir, "global_metadata.json")
        data = {
            "databases": self.global_databases_meta,
            "updated_at": utc_isoformat(),
            "version": "2.0",
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _normalize_global_metadata(self) -> None:
        """规范化全局元数据中的时间戳字段。"""
        for meta in self.global_databases_meta.values():
            if "created_at" not in meta:
                continue
            try:
                dt_value = coerce_any_to_utc_datetime(meta["created_at"])
                if dt_value:
                    meta["created_at"] = utc_isoformat(dt_value)
            except Exception as exc:
                logger.warning(
                    f"Failed to normalize database metadata timestamp {meta['created_at']!r}: {exc}"
                )

    # ------------------------------------------------------------------
    # 实例管理
    # ------------------------------------------------------------------

    def _initialize_existing_kbs(self) -> None:
        """为已记录的每种知识库类型预创建实例。"""
        kb_types_in_use = {
            meta.get("kb_type", _DEFAULT_KB_TYPE)
            for meta in self.global_databases_meta.values()
        }
        for kb_type in kb_types_in_use:
            try:
                self._get_or_create_kb_instance(kb_type)
            except Exception as e:
                logger.error(f"Failed to initialize {kb_type} knowledge base: {e}")

    def _get_or_create_kb_instance(self, kb_type: str) -> KnowledgeBase:
        """获取或懒创建指定类型的知识库实例（单例，按类型缓存）。"""
        if kb_type in self.kb_instances:
            return self.kb_instances[kb_type]

        kb_work_dir = os.path.join(self.work_dir, f"{kb_type}_data")
        kb_instance = KnowledgeBaseFactory.create(kb_type, kb_work_dir)
        self.kb_instances[kb_type] = kb_instance
        logger.info(f"Created {kb_type} knowledge base instance")
        return kb_instance

    def _get_kb_for_database(self, db_id: str) -> KnowledgeBase:
        """根据数据库 ID 获取对应的知识库实例。

        Raises:
            KBNotFoundError: 数据库不存在或其类型不受支持。
        """
        if db_id not in self.global_databases_meta:
            raise KBNotFoundError(f"Database {db_id} not found")

        kb_type = self.global_databases_meta[db_id].get("kb_type", _DEFAULT_KB_TYPE)
        if not KnowledgeBaseFactory.is_type_supported(kb_type):
            raise KBNotFoundError(f"Unsupported knowledge base type: {kb_type}")

        return self._get_or_create_kb_instance(kb_type)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def get_kb(self, db_id: str) -> KnowledgeBase:
        """通过数据库 ID 获取底层知识库实例（兼容性接口）。"""
        return self._get_kb_for_database(db_id)

    def get_databases(self) -> dict:
        """获取所有数据库信息。"""
        all_databases: list[dict] = []
        for kb_instance in self.kb_instances.values():
            all_databases.extend(kb_instance.get_databases()["databases"])
        return {"databases": all_databases}

    async def create_database(
        self,
        database_name: str,
        description: str,
        kb_type: str,
        embed_info: dict | None = None,
        **kwargs,
    ) -> dict:
        """创建数据库。

        Args:
            database_name: 数据库名称。
            description:   数据库描述。
            kb_type:       知识库类型。
            embed_info:    嵌入模型信息。
            **kwargs:      传入底层 create_database 的额外参数（如 chunk_size）。

        Raises:
            ValueError: 不支持的知识库类型。
        """
        if not KnowledgeBaseFactory.is_type_supported(kb_type):
            available_types = list(KnowledgeBaseFactory.get_available_types().keys())
            raise ValueError(
                f"Unsupported knowledge base type: {kb_type}. Available types: {available_types}"
            )

        kb_instance = self._get_or_create_kb_instance(kb_type)
        db_info = kb_instance.create_database(database_name, description, embed_info, **kwargs)
        db_id = db_info["db_id"]

        async with self._metadata_lock:
            self.global_databases_meta[db_id] = {
                "name": database_name,
                "description": description,
                "kb_type": kb_type,
                "created_at": utc_isoformat(),
                "additional_params": kwargs.copy(),
            }
            self._save_global_metadata()

        logger.info(f"Created {kb_type} database: {database_name} ({db_id})")
        return db_info

    async def delete_database(self, db_id: str) -> dict:
        """删除数据库。"""
        try:
            kb_instance = self._get_kb_for_database(db_id)
            result = kb_instance.delete_database(db_id)
        except KBNotFoundError as e:
            logger.warning(f"Database {db_id} not found during deletion: {e}")
            return {"message": "删除成功"}

        async with self._metadata_lock:
            self.global_databases_meta.pop(db_id, None)
            self._save_global_metadata()

        return result

    async def add_content(self, db_id: str, items: list[str], params: dict | None = None) -> list[dict]:
        """添加内容（文件/URL）。"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.add_content(db_id, items, params or {})

    async def add_processed_content(self, db_id: str, data: dict | None = None) -> list[dict]:
        """添加已处理好的内容（如 Markdown 切片）。"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.add_processed_content(db_id, data)

    async def add_image_embeddings(self, db_id: str, items: list[str], params: dict | None = None) -> list[dict]:
        """添加图片嵌入。"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.add_image_embeddings(db_id, items, params or {})

    async def aquery(self, query_text: Union[str, List[str]], db_id: str, **kwargs) -> list[dict]:
        """异步查询知识库。"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.aquery(db_id, query_text, **kwargs)

    async def export_data(self, db_id: str, format: str = "zip", **kwargs) -> str:
        """导出知识库数据。"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.export_data(db_id, format=format, **kwargs)

    def query(self, query_text: str, db_id: str, **kwargs) -> str:
        """同步查询知识库（兼容性方法）。"""
        kb_instance = self._get_kb_for_database(db_id)
        return kb_instance.query(query_text, db_id, **kwargs)

    def get_database_info(self, db_id: str) -> dict | None:
        """获取数据库详细信息（含 additional_params）。"""
        try:
            kb_instance = self._get_kb_for_database(db_id)
            db_info = kb_instance.get_database_info(db_id)
        except KBNotFoundError:
            return None

        if db_info and db_id in self.global_databases_meta:
            additional_params = self.global_databases_meta[db_id].get("additional_params", {})
            if additional_params:
                db_info["additional_params"] = additional_params

        return db_info

    async def delete_file(self, db_id: str, file_id: str) -> None:
        """删除文件。"""
        kb_instance = self._get_kb_for_database(db_id)
        await kb_instance.delete_file(db_id, file_id)

    async def get_file_basic_info(self, db_id: str, file_id: str) -> dict:
        """获取文件基本信息（仅元数据）。"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.get_file_basic_info(db_id, file_id)

    async def get_file_content(self, db_id: str, file_id: str) -> dict:
        """获取文件内容信息（chunks 和 lines）。"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.get_file_content(db_id, file_id)

    async def get_file_info(self, db_id: str, file_id: str) -> dict:
        """获取文件完整信息（基本信息 + 内容信息）- 保持向后兼容。"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.get_file_info(db_id, file_id)

    def get_db_upload_path(self, db_id: str | None = None) -> str:
        """获取数据库上传路径。"""
        if db_id:
            try:
                kb_instance = self._get_kb_for_database(db_id)
                return kb_instance.get_db_upload_path(db_id)
            except KBNotFoundError:
                pass

        general_uploads = os.path.join(self.work_dir, "uploads")
        os.makedirs(general_uploads, exist_ok=True)
        return general_uploads

    def file_existed_in_db(self, db_id: str | None, content_hash: str | None) -> bool:
        """检查指定数据库中是否存在相同内容哈希的文件。"""
        if not db_id or not content_hash:
            return False
        try:
            kb_instance = self._get_kb_for_database(db_id)
        except KBNotFoundError:
            return False

        return any(
            file_info.get("database_id") == db_id and file_info.get("content_hash") == content_hash
            for file_info in kb_instance.files_meta.values()
        )

    async def update_database(
        self,
        db_id: str,
        name: str,
        description: str,
        additional_params: dict | None = None,
    ) -> dict:
        """更新数据库名称、描述和附加参数。"""
        kb_instance = self._get_kb_for_database(db_id)
        result = kb_instance.update_database(db_id, name, description)

        async with self._metadata_lock:
            if db_id in self.global_databases_meta:
                self.global_databases_meta[db_id]["name"] = name
                self.global_databases_meta[db_id]["description"] = description
                if additional_params is not None:
                    self.global_databases_meta[db_id]["additional_params"] = additional_params
                self._save_global_metadata()

        return result

    def list_database_documents(self, db_id: str) -> list[dict]:
        """列出某个知识库下的所有文档（按创建时间倒序）。"""
        kb_instance = self._get_kb_for_database(db_id)
        documents = [
            {
                "file_id": file_id,
                "filename": file_info.get("filename", ""),
                "path": file_info.get("path", ""),
                "file_type": file_info.get("file_type", ""),
                "status": file_info.get("status", ""),
                "created_at": file_info.get("created_at", ""),
            }
            for file_id, file_info in kb_instance.files_meta.items()
            if file_info.get("database_id") == db_id
        ]
        documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return documents

    def register_file_records(self, db_id: str, records: list[dict]) -> None:
        """批量注册文件记录（每条 record 必须包含 file_id 字段）。"""
        kb_instance = self._get_kb_for_database(db_id)
        for record in records:
            file_id = record.get("file_id")
            if not file_id:
                continue
            kb_instance.files_meta[file_id] = record
        kb_instance._save_metadata()

    async def rebuild_database(self, db_id: str, params: dict | None = None) -> dict:
        """使用数据库内已记录的文件路径重新构建索引。"""
        kb_instance = self._get_kb_for_database(db_id)
        docs = self.list_database_documents(db_id)
        file_paths = [d.get("path") for d in docs if d.get("path")]
        if not file_paths:
            return {"status": "skipped", "message": "知识库中没有可重建文件", "db_id": db_id}

        for d in docs:
            try:
                await kb_instance.delete_file(db_id, d["file_id"])
            except Exception as e:
                logger.warning(f"Failed to delete old indexed file {d['file_id']} during rebuild: {e}")

        rebuild_params = {**(params or {}), "content_type": (params or {}).get("content_type", "file")}
        result = await kb_instance.add_content(db_id, file_paths, rebuild_params)
        return {
            "status": "success",
            "db_id": db_id,
            "reindexed_files": len(result),
            "params": rebuild_params,
        }

    # ------------------------------------------------------------------
    # 管理器元信息
    # ------------------------------------------------------------------

    def get_supported_kb_types(self) -> dict[str, dict]:
        """获取所有支持的知识库类型。"""
        return KnowledgeBaseFactory.get_available_types()

    def get_kb_instance_info(self) -> dict[str, dict]:
        """获取各知识库实例的统计信息。"""
        return {
            kb_type: {
                "work_dir": kb_instance.work_dir,
                "database_count": len(kb_instance.databases_meta),
                "file_count": len(kb_instance.files_meta),
            }
            for kb_type, kb_instance in self.kb_instances.items()
        }

    def get_statistics(self) -> dict:
        """获取全局统计信息。"""
        stats: dict = {
            "total_databases": len(self.global_databases_meta),
            "kb_types": {},
            "total_files": 0,
        }
        for db_meta in self.global_databases_meta.values():
            kb_type = db_meta.get("kb_type", _DEFAULT_KB_TYPE)
            stats["kb_types"][kb_type] = stats["kb_types"].get(kb_type, 0) + 1

        for kb_instance in self.kb_instances.values():
            stats["total_files"] += len(kb_instance.files_meta)

        return stats
