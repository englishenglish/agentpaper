import hashlib
import os
import time
from pathlib import Path
import json

from src.core.config import config
from src.utils import hashstr
from src.utils.datetime_utils import utc_isoformat
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

def validate_file_path(file_path: str, db_id: str = None) -> str:
    """
    验证文件路径安全性，防止路径遍历攻击

    Args:
        file_path: 要验证的文件路径
        db_id: 数据库ID，用于获取知识库特定的上传目录

    Returns:
        str: 规范化后的安全路径

    Raises:
        ValueError: 如果路径不安全
    """
    try:
        # 规范化路径
        normalized_path = os.path.abspath(os.path.realpath(file_path))

        # 获取允许的根目录
        from src.rag import knowledge_base

        allowed_dirs = [
            os.path.abspath(os.path.realpath(config.get("SAVE_DIR"))),
        ]

        # 如果指定了db_id，添加知识库特定的上传目录
        if db_id:
            try:
                allowed_dirs.append(os.path.abspath(os.path.realpath(knowledge_base.get_db_upload_path(db_id))))
            except Exception:
                # 如果无法获取db路径，使用通用上传目录
                allowed_dirs.append(
                    os.path.abspath(os.path.realpath(os.path.join(config.get("SAVE_DIR"), "database", "uploads")))
                )

        # 检查路径是否在允许的目录内
        is_safe = False
        for allowed_dir in allowed_dirs:
            try:
                if normalized_path.startswith(allowed_dir):
                    is_safe = True
                    break
            except Exception:
                continue

        if not is_safe:
            logger.warning(f"Path traversal attempt detected: {file_path} (normalized: {normalized_path})")
            raise ValueError(f"Access denied: Invalid file path: {file_path}")

        return normalized_path

    except Exception as e:
        logger.error(f"Path validation failed for {file_path}: {e}")
        raise ValueError(f"Invalid file path: {file_path}")


def calculate_content_hash(data: bytes | bytearray | str | os.PathLike[str] | Path) -> str:
    """
    计算文件内容的 SHA-256 哈希值。

    Args:
        data: 文件内容的二进制数据或文件路径

    Returns:
        str: 十六进制哈希值
    """
    sha256 = hashlib.sha256()

    if isinstance(data, (bytes, bytearray)):
        sha256.update(data)
        return sha256.hexdigest()

    if isinstance(data, (str, os.PathLike, Path)):
        path = Path(data)
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    raise TypeError(f"Unsupported data type for hashing: {type(data)!r}")


def prepare_item_metadata(item: str, content_type: str, db_id: str) -> dict:
    """
    准备文件或URL的元数据
    """
    if content_type == "file" or content_type == "json":
        file_path = Path(item)
        file_id = f"file_{hashstr(str(file_path) + str(time.time()), 6)}"
        file_type = file_path.suffix.lower().replace(".", "")
        filename = file_path.name
        item_path = os.path.relpath(file_path, Path.cwd())
        content_hash = None
        try:
            if file_path.exists():
                content_hash = calculate_content_hash(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to calculate content hash for {file_path}: {exc}")
    else:  # URL
        file_id = f"url_{hashstr(item + str(time.time()), 6)}"
        file_type = "url"
        filename = f"webpage_{hashstr(item, 6)}.md"
        item_path = item
        content_hash = None

    return {
        "database_id": db_id,
        "filename": filename,
        "path": item_path,
        "file_type": file_type,
        "status": "processing",
        "created_at": utc_isoformat(),
        "file_id": file_id,
        "content_hash": content_hash,
    }


def get_embedding_config(embed_info: dict) -> dict:
    """
    获取嵌入模型配置

    Args:
        embed_info: 嵌入信息字典

    Returns:
        dict: 标准化的嵌入配置
    """
    config_dict = {}

    try:
        if not embed_info:
            embeding_dic = config.get("embedding-model")
            embedding_provider = embeding_dic.get("model-provider")
            provider_dic = config.get(embedding_provider)
            
            embed_info = {
                "name":embeding_dic.get("model"),
                "dimension": embeding_dic.get("dimension"),
                "base_url": provider_dic.get("base_url"),
                "api_key": provider_dic.get("api_key"), 
            }
        config_dict["model"] = embed_info["name"]
        config_dict["api_key"] = embed_info["api_key"]
        config_dict["base_url"] = embed_info["base_url"]
        config_dict["dimension"] = embed_info.get("dimension", 1024)


    except Exception as e:
        logger.error(f"Error in get_embedding_config: {e}, {embed_info}")
        raise ValueError(f"Error in get_embedding_config: {e}")

    logger.debug(f"Embedding config: {config_dict}")
    return config_dict

def validate_img_embedding_file(file_path: str) -> bool:
    """校验 JSON 是否为图片嵌入列表（name / image_url / detail_url / description）。"""
    file_path_obj = Path(file_path)
    if file_path_obj.suffix.lower() != ".json" or not file_path_obj.exists():
        return False
    try:
        with open(file_path, encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, list):
        return False
    required = {"name", "image_url", "detail_url", "description"}
    for item in payload:
        if not isinstance(item, dict):
            return False
        if required - set(item.keys()):
            return False
        for k in required:
            if not isinstance(item[k], str):
                return False
        if item["image_url"] and not item["image_url"].startswith(("http://", "https://")):
            return False
        if item["detail_url"] and not item["detail_url"].startswith(("http://", "https://")):
            return False
    return True