"""
embedding.py — 语义向量编码器（替代 graph_store.py 中的伪哈希嵌入）

支持两种后端：
  1. SentenceTransformer（默认，本地模型）
  2. OpenAI / 兼容 API（通过 api_base + api_key 注入）

接口统一：
  get_embedding(text)        → list[float]
  get_embeddings(texts)      → list[list[float]]
"""
from __future__ import annotations

import math
from typing import Any

from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

# 默认轻量模型（~22MB，384 维）
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class VectorEmbedder:
    """
    语义向量编码器，封装 SentenceTransformer 或 OpenAI Embeddings API。

    依赖注入友好：所有可变配置通过 __init__ 传入，不从全局 config 直接读取，
    便于在 Agent 容器或测试中替换。

    Args:
        model_name:   SentenceTransformer 模型名（本地或 HuggingFace Hub）。
        backend:      "sentence_transformers" | "openai"。
        api_client:   已初始化的 OpenAI 客户端（仅 backend="openai" 时使用）。
        api_model:    OpenAI Embedding 模型名，如 "text-embedding-3-small"。
        device:       torch device，如 "cpu" / "cuda"（仅 sentence_transformers）。
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        backend: str = "sentence_transformers",
        api_client: Any = None,
        api_model: str = "text-embedding-3-small",
        device: str = "cpu",
    ) -> None:
        self.backend = backend
        self._model: Any = None
        self._api_client = api_client
        self._api_model = api_model

        if backend == "sentence_transformers":
            self._model = self._load_st_model(model_name, device)
        elif backend == "openai":
            if api_client is None:
                raise ValueError("backend='openai' 需要传入 api_client 实例")
        else:
            raise ValueError(f"不支持的 backend：{backend!r}")

        logger.info(
            f"VectorEmbedder 初始化完成 | backend={backend} | "
            f"model={model_name if backend != 'openai' else api_model}"
        )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def get_embedding(self, text: str) -> list[float]:
        """对单个字符串编码，返回归一化后的 list[float]。"""
        if not text or not text.strip():
            return self._zero_vector()
        return self.get_embeddings([text])[0]

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量编码，返回 list[list[float]]，每条向量已 L2 归一化。"""
        if not texts:
            return []
        cleaned = [t.strip() if t else "" for t in texts]

        if self.backend == "sentence_transformers":
            return self._encode_st(cleaned)
        elif self.backend == "openai":
            return self._encode_openai(cleaned)
        return [self._zero_vector() for _ in cleaned]

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _load_st_model(self, model_name: str, device: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer(model_name, device=device)
            return model
        except ImportError:
            logger.warning(
                "sentence-transformers 未安装，回退到零向量模式。"
                "请执行：pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.warning(f"加载 SentenceTransformer 模型失败（{model_name}）：{e}，回退到零向量模式")
            return None

    def _encode_st(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            return [self._zero_vector() for _ in texts]
        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.warning(f"SentenceTransformer 编码失败：{e}")
            return [self._zero_vector() for _ in texts]

    def _encode_openai(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._api_client.embeddings.create(
                model=self._api_model,
                input=texts,
            )
            vecs = [item.embedding for item in response.data]
            return [self._l2_normalize(v) for v in vecs]
        except Exception as e:
            logger.warning(f"OpenAI Embedding API 调用失败：{e}")
            return [self._zero_vector() for _ in texts]

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _zero_vector(dim: int = 384) -> list[float]:
        return [0.0] * dim


def embedding_cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    余弦相似度计算（向量已归一化时等价于点积）。
    返回值映射到 [0, 1]。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, (dot + 1.0) / 2.0))
