from __future__ import annotations

from src.knowledge.base import KBNotFoundError, KnowledgeBase
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


class KnowledgeBaseFactory:
    """知识库工厂类，负责创建不同类型的知识库实例。"""

    # 注册的知识库类型映射 {kb_type: kb_class}
    _kb_types: dict[str, type[KnowledgeBase]] = {}

    # 每种类型的默认配置
    _default_configs: dict[str, dict] = {}

    @classmethod
    def register(
        cls,
        kb_type: str,
        kb_class: type[KnowledgeBase],
        default_config: dict | None = None,
    ) -> None:
        """
        注册知识库类型。

        Args:
            kb_type:        知识库类型标识。
            kb_class:       知识库实现类，必须继承自 KnowledgeBase。
            default_config: 该类型的默认初始化配置。
        """
        if not issubclass(kb_class, KnowledgeBase):
            raise ValueError("Knowledge base class must inherit from KnowledgeBase")
        cls._kb_types[kb_type] = kb_class
        cls._default_configs[kb_type] = default_config or {}
        logger.info(f"Registered knowledge base type: {kb_type}")

    @classmethod
    def create(cls, kb_type: str, work_dir: str, **kwargs) -> KnowledgeBase:
        """
        创建知识库实例。

        Args:
            kb_type:  知识库类型。
            work_dir: 工作目录。
            **kwargs: 传递给知识库构造函数的额外参数（会覆盖默认配置）。

        Returns:
            已初始化的知识库实例。

        Raises:
            KBNotFoundError: 未知的知识库类型。
        """
        if kb_type not in cls._kb_types:
            available_types = list(cls._kb_types.keys())
            raise KBNotFoundError(
                f"Unknown knowledge base type: {kb_type}. Available types: {available_types}"
            )

        kb_class = cls._kb_types[kb_type]
        init_config = cls._default_configs[kb_type].copy()
        init_config.update(kwargs)

        try:
            instance = kb_class(work_dir, **init_config)
            logger.info(f"Created {kb_type} knowledge base instance at {work_dir}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create {kb_type} knowledge base: {e}")
            raise

    @classmethod
    def get_available_types(cls) -> dict[str, dict]:
        """返回所有已注册的知识库类型及其描述。"""
        return {
            kb_type: {
                "class_name": kb_class.__name__,
                "description": kb_class.__doc__ or "",
                "default_config": cls._default_configs[kb_type],
            }
            for kb_type, kb_class in cls._kb_types.items()
        }

    @classmethod
    def is_type_supported(cls, kb_type: str) -> bool:
        """检查是否支持指定的知识库类型。"""
        return kb_type in cls._kb_types

    @classmethod
    def get_default_config(cls, kb_type: str) -> dict:
        """返回指定类型的默认配置副本。"""
        return cls._default_configs.get(kb_type, {}).copy()
