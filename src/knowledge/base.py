from __future__ import annotations

import json
import os
import shutil
import threading
from abc import ABC, abstractmethod
from typing import Any, List, Union

from src.utils.datetime_utils import coerce_any_to_utc_datetime, utc_isoformat
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


class KnowledgeBaseException(Exception):
    """知识库统一异常基类"""


class KBNotFoundError(KnowledgeBaseException):
    """知识库不存在错误"""


class KBOperationError(KnowledgeBaseException):
    """知识库操作错误"""


class KnowledgeBase(ABC):
    """知识库抽象基类，定义统一接口"""

    # 类级别的处理队列，跟踪所有正在处理的文件
    _processing_files: set[str] = set()
    _processing_lock: threading.Lock | None = None

    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir
        self.databases_meta: dict[str, dict] = {}
        self.files_meta: dict[str, dict] = {}

        if KnowledgeBase._processing_lock is None:
            KnowledgeBase._processing_lock = threading.Lock()

        os.makedirs(work_dir, exist_ok=True)
        self._load_metadata()
        self._normalize_metadata_state()

    # ------------------------------------------------------------------
    # 时间戳规范化
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_timestamp(value: Any) -> str | None:
        """将持久化时间戳统一转换为 UTC ISO 字符串。"""
        try:
            dt_value = coerce_any_to_utc_datetime(value)
        except (TypeError, ValueError) as exc:
            logger.warning(f"Invalid timestamp encountered: {value!r} ({exc})")
            return None
        if not dt_value:
            return None
        return utc_isoformat(dt_value)

    def _normalize_metadata_state(self) -> None:
        """确保内存中的元数据使用规范化的时间戳格式。"""
        for meta in self.databases_meta.values():
            if "created_at" in meta:
                normalized = self._normalize_timestamp(meta.get("created_at"))
                if normalized:
                    meta["created_at"] = normalized

        for file_info in self.files_meta.values():
            if "created_at" in file_info:
                normalized = self._normalize_timestamp(file_info.get("created_at"))
                if normalized:
                    file_info["created_at"] = normalized

    # ------------------------------------------------------------------
    # 抽象接口
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def kb_type(self) -> str:
        """知识库类型标识"""

    @abstractmethod
    async def _create_kb_instance(self, db_id: str, config: dict) -> Any:
        """创建底层知识库实例"""

    @abstractmethod
    async def _initialize_kb_instance(self, instance: Any) -> None:
        """初始化底层知识库实例"""

    @abstractmethod
    async def add_processed_content(self, db_id: str, data: dict | None = None) -> list[dict]:
        """添加内容（已处理好的数据）"""

    @abstractmethod
    async def add_content(self, db_id: str, items: list[str], params: dict | None = None) -> list[dict]:
        """添加内容（文件/URL）"""

    @abstractmethod
    async def add_image_embeddings(self, db_id: str, items: list[str], params: dict | None = None) -> list[dict]:
        """添加图片嵌入"""

    @abstractmethod
    async def aquery(self, db_id: str, query_text: Union[str, List[str]] = None, **kwargs) -> list[dict]:
        """异步查询知识库"""

    @abstractmethod
    async def delete_file(self, db_id: str, file_id: str) -> None:
        """删除文件"""

    @abstractmethod
    async def get_file_basic_info(self, db_id: str, file_id: str) -> dict:
        """获取文件基本信息（仅元数据）"""

    @abstractmethod
    async def get_file_content(self, db_id: str, file_id: str) -> dict:
        """获取文件内容信息（chunks 和 lines）"""

    @abstractmethod
    async def get_file_info(self, db_id: str, file_id: str) -> dict:
        """获取文件完整信息（基本信息 + 内容信息）- 保持向后兼容"""

    # ------------------------------------------------------------------
    # 具体公共方法
    # ------------------------------------------------------------------

    def create_database(
        self,
        database_name: str,
        description: str,
        embed_info: dict | None = None,
        llm_info: dict | None = None,
        **kwargs,
    ) -> dict:
        """创建数据库并返回数据库信息字典。"""
        from src.utils import hashstr

        db_id = f"kb_{hashstr(database_name, with_salt=True)}"
        self.databases_meta[db_id] = {
            "name": database_name,
            "description": description,
            "kb_type": self.kb_type,
            "embed_info": embed_info,
            "llm_info": llm_info,
            "metadata": kwargs,
            "created_at": utc_isoformat(),
        }
        self._save_metadata()

        working_dir = os.path.join(self.work_dir, db_id)
        os.makedirs(working_dir, exist_ok=True)

        db_dict = self.databases_meta[db_id].copy()
        db_dict["db_id"] = db_id
        db_dict["files"] = {}
        return db_dict

    def delete_database(self, db_id: str) -> dict:
        """删除数据库及其工作目录，返回操作结果。"""
        if db_id in self.databases_meta:
            files_to_delete = [
                fid for fid, finfo in self.files_meta.items()
                if finfo.get("database_id") == db_id
            ]
            for file_id in files_to_delete:
                del self.files_meta[file_id]
            del self.databases_meta[db_id]
            self._save_metadata()

        working_dir = os.path.join(self.work_dir, db_id)
        if os.path.exists(working_dir):
            try:
                shutil.rmtree(working_dir)
            except Exception as e:
                logger.error(f"Error deleting working directory {working_dir}: {e}")

        return {"message": "删除成功"}

    async def export_data(self, db_id: str, format: str = "zip", **kwargs) -> str:
        """导出知识库数据（子类可覆盖实现）。"""
        return ""

    def query(self, query_text: str, db_id: str, **kwargs) -> list[dict]:
        """同步查询知识库（兼容性方法，新代码请使用 aquery）。"""
        import asyncio

        logger.warning("query() is deprecated, use aquery() instead")
        return asyncio.run(self.aquery(query_text, db_id, **kwargs))

    def get_database_info(self, db_id: str) -> dict | None:
        """获取数据库详细信息，包含文件列表（按创建时间倒序）。"""
        if db_id not in self.databases_meta:
            return None

        meta = self.databases_meta[db_id].copy()
        meta["db_id"] = db_id
        self._check_and_fix_processing_status(db_id)
        meta["files"] = self._build_db_files(db_id)
        meta["row_count"] = len(meta["files"])
        meta["status"] = "已连接"
        return meta

    def get_databases(self) -> dict:
        """获取所有数据库信息列表。"""
        databases = []
        for db_id, meta in self.databases_meta.items():
            self._check_and_fix_processing_status(db_id)
            db_dict = meta.copy()
            db_dict["db_id"] = db_id
            db_dict["files"] = self._build_db_files(db_id)
            db_dict["row_count"] = len(db_dict["files"])
            db_dict["status"] = "已连接"
            databases.append(db_dict)
        return {"databases": databases}

    def get_db_upload_path(self, db_id: str | None = None) -> str:
        """获取数据库上传路径，不传 db_id 时返回通用路径。"""
        if db_id:
            uploads_folder = os.path.join(self.work_dir, db_id, "uploads")
            os.makedirs(uploads_folder, exist_ok=True)
            return uploads_folder

        general_uploads = os.path.join(self.work_dir, "uploads")
        os.makedirs(general_uploads, exist_ok=True)
        return general_uploads

    def update_database(self, db_id: str, name: str, description: str) -> dict:
        """更新数据库名称和描述。"""
        if db_id not in self.databases_meta:
            raise ValueError(f"数据库 {db_id} 不存在")
        self.databases_meta[db_id]["name"] = name
        self.databases_meta[db_id]["description"] = description
        self._save_metadata()
        return self.get_database_info(db_id)

    # ------------------------------------------------------------------
    # 处理队列（类方法）
    # ------------------------------------------------------------------

    @classmethod
    def _add_to_processing_queue(cls, file_id: str) -> None:
        with cls._processing_lock:
            cls._processing_files.add(file_id)
            logger.debug(f"Added file {file_id} to processing queue")

    @classmethod
    def _remove_from_processing_queue(cls, file_id: str) -> None:
        with cls._processing_lock:
            cls._processing_files.discard(file_id)
            logger.debug(f"Removed file {file_id} from processing queue")

    @classmethod
    def _is_file_in_processing_queue(cls, file_id: str) -> bool:
        with cls._processing_lock:
            return file_id in cls._processing_files

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _build_db_files(self, db_id: str) -> dict:
        """构建指定数据库的文件列表（已规范化时间戳，按创建时间倒序）。"""
        db_files = {}
        for file_id, file_info in self.files_meta.items():
            if file_info.get("database_id") != db_id:
                continue
            created_at = self._normalize_timestamp(file_info.get("created_at"))
            db_files[file_id] = {
                "file_id": file_id,
                "filename": file_info.get("filename", ""),
                "path": file_info.get("path", ""),
                "type": file_info.get("file_type", ""),
                "status": file_info.get("status", "done"),
                "created_at": created_at,
            }
        return dict(
            sorted(
                db_files.items(),
                key=lambda item: item[1].get("created_at") or "",
                reverse=True,
            )
        )

    def _check_and_fix_processing_status(self, db_id: str) -> None:
        """将状态为 processing 但不在队列中的文件标记为 error。"""
        try:
            status_changed = False
            for file_id, file_info in self.files_meta.items():
                if (
                    file_info.get("database_id") == db_id
                    and file_info.get("status") == "processing"
                    and not self._is_file_in_processing_queue(file_id)
                ):
                    logger.warning(
                        f"File {file_id} has processing status but is not in processing queue, marking as error"
                    )
                    file_info["status"] = "error"
                    file_info["error"] = "Processing interrupted - file not found in processing queue"
                    status_changed = True

            if status_changed:
                self._save_metadata()
                logger.info(f"Fixed processing status for database {db_id}")
        except Exception as e:
            logger.error(f"Error checking processing status for database {db_id}: {e}")

    def _load_metadata(self) -> None:
        """从磁盘加载元数据。"""
        meta_file = os.path.join(self.work_dir, f"metadata_{self.kb_type}.json")
        if not os.path.exists(meta_file):
            return
        try:
            with open(meta_file, encoding="utf-8") as f:
                data = json.load(f)
            self.databases_meta = data.get("databases", {})
            self.files_meta = data.get("files", {})
            logger.info(f"Loaded {self.kb_type} metadata for {len(self.databases_meta)} databases")
        except Exception as e:
            logger.error(f"Failed to load {self.kb_type} metadata: {e}")

    def _save_metadata(self) -> None:
        """将元数据持久化到磁盘。"""
        self._normalize_metadata_state()
        meta_file = os.path.join(self.work_dir, f"metadata_{self.kb_type}.json")
        try:
            data = {
                "databases": self.databases_meta,
                "files": self.files_meta,
                "kb_type": self.kb_type,
                "updated_at": utc_isoformat(),
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save {self.kb_type} metadata: {e}")
