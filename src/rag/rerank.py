from FlagEmbedding import FlagReranker


class BGEReranker:
    def __init__(self, model_name_or_path="BAAI/bge-reranker-v2-m3", use_fp16=True):
        self.reranker = FlagReranker(model_name_or_path, use_fp16=use_fp16)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """对召回的文档进行交叉打分"""
        pairs = [[query, doc] for doc in docs]
        scores = self.reranker.compute_score(pairs, normalize=True)
        # 如果只有一条数据，API 返回的是 float，将其统一转为 list
        if isinstance(scores, float):
            scores = [scores]
        return scores