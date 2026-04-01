"""
db_client.py — Neo4j 图数据库客户端

职责：
  - 管理 Neo4j 连接生命周期（connect / close / context manager）
  - sync_graph()      : 将内存图谱全量同步到 Neo4j
  - get_local_subgraph(): 通过 Cypher 直接在 Neo4j 侧做 BFS，
                          替代原先 Python 字典 BFS 遍历，降低 I/O 传输量

依赖注入：
  Neo4jGraphClient(uri, user, password)
  — 所有连接参数通过构造函数注入，不从全局 config 读取，便于多实例/测试替换。

Cypher 优化说明（Optimization 2 & 3）：
  get_local_subgraph 使用单条 Cypher 查询，在数据库侧完成：
    1. 以 top_k_entities 为起点进行 1-2 跳 BFS（MATCH (n)-[r*1..2]->(m)）
    2. 直接返回（源节点标签，关系类型，目标节点标签，目标节点类型）四元组
    3. 避免 Python 层面的循环 BFS，时间复杂度从 O(V+E) 降至网络一次 RTT
"""
from __future__ import annotations

from typing import Any

from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

try:
    from neo4j import GraphDatabase as _Neo4jDriver  # type: ignore
    _NEO4J_AVAILABLE = True
except ImportError:
    _Neo4jDriver = None  # type: ignore
    _NEO4J_AVAILABLE = False

# 关系语义化标签（与 graph_store.py 保持一致，用于 subgraph 文本渲染）
_RELATION_LABELS: dict[str, str] = {
    "proposes":        "proposes",
    "improves":        "improves upon",
    "uses":            "uses",
    "evaluates_on":    "evaluated on",
    "compared_with":   "compared with",
    "achieves":        "achieves",
    "applied_to":      "applied to",
    "related_to":      "related to",
    "has_experiment":  "has experiment",
    "uses_dataset":    "uses dataset",
    "measures":        "measures",
    "produces":        "produces result",
    "has_contribution":"contributes",
    "solves":          "solves",
    "cites":           "cites",
    "extends":         "extends",
}


class Neo4jGraphClient:
    """
    Neo4j 图数据库客户端。

    支持 with 语句（上下文管理器）和手动 close()。

    Args:
        uri:      Neo4j bolt URI，如 "neo4j://localhost:7687"。
        user:     数据库用户名。
        password: 数据库密码。

    Raises:
        RuntimeError: 若 neo4j Python 驱动未安装。
    """

    def __init__(
        self,
        uri: str = "neo4j://localhost:7687",
        user: str = "neo4j",
        password: str = "123456",
    ) -> None:
        if not _NEO4J_AVAILABLE:
            raise RuntimeError(
                "neo4j 驱动未安装，请执行：pip install neo4j"
            )
        self._uri      = uri
        self._user     = user
        self._password = password
        self._driver   = None
        self._connect()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        try:
            self._driver = _Neo4jDriver.driver(
                self._uri, auth=(self._user, self._password)
            )
            logger.info(f"Neo4j 连接成功：{self._uri}")
        except Exception as e:
            logger.error(f"Neo4j 连接失败（{self._uri}）：{e}")
            raise

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j 连接已关闭")

    def __enter__(self) -> "Neo4jGraphClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 图谱同步
    # ------------------------------------------------------------------

    def sync_graph(self, graph_data: dict[str, Any], db_id: str) -> None:
        """
        将内存图谱全量同步到 Neo4j（先清空再写入）。

        同步策略：
          1. DETACH DELETE 当前 db_id 下所有节点（原子清理）
          2. MERGE 所有节点（以 id + db_id 为复合唯一键）
          3. MERGE 所有关系（以 type 为关系唯一键）

        Args:
            graph_data: 图谱字典（nodes / edges）。
            db_id:      图谱标识符，用于多租户隔离。
        """
        if self._driver is None:
            raise RuntimeError("Neo4j 未连接，请先调用 _connect()")

        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])

        def _sync(tx: Any) -> None:
            # 清理旧数据
            tx.run(
                "MATCH (n:Entity {db_id: $db_id}) DETACH DELETE n",
                db_id=db_id,
            )
            # 写入节点
            for node in nodes.values():
                tx.run(
                    """
                    MERGE (n:Entity {id: $id, db_id: $db_id})
                    SET n.label       = $label,
                        n.type        = $type,
                        n.norm_label  = $norm_label,
                        n.description = $description
                    """,
                    id=node.get("id"),
                    db_id=db_id,
                    label=node.get("label", ""),
                    type=node.get("type", ""),
                    norm_label=node.get("norm_label", ""),
                    description=node.get("description", ""),
                )
            # 写入关系
            for edge in edges:
                tx.run(
                    """
                    MATCH (s:Entity {id: $src, db_id: $db_id})
                    MATCH (t:Entity {id: $dst, db_id: $db_id})
                    MERGE (s)-[r:RELATION {type: $type}]->(t)
                    SET r.weight = $weight
                    """,
                    src=edge.get("source"),
                    dst=edge.get("target"),
                    type=edge.get("type"),
                    weight=float(edge.get("weight", 1.0) or 1.0),
                    db_id=db_id,
                )

        with self._driver.session() as session:
            session.execute_write(_sync)

        logger.info(
            f"图谱已同步到 Neo4j（db_id={db_id}，"
            f"节点={len(nodes)}，边={len(edges)}）"
        )

    # ------------------------------------------------------------------
    # 局部子图查询（Cypher BFS，替代 Python 字典 BFS）
    # ------------------------------------------------------------------

    def get_local_subgraph(
        self,
        top_k_entities: list[str],
        db_id: str,
        max_hops: int = 2,
        max_triples: int = 40,
    ) -> str:
        """
        通过 Cypher 在 Neo4j 侧查询以 top_k_entities 为起点的局部子图，
        转换为结构化可读知识块（供 LLM 推理）。

        相比原始 Python 字典 BFS：
          - 数据库侧完成图遍历，避免全量节点加载到内存
          - 单次网络请求替代多轮 Python 循环
          - 时间复杂度从 O(V+E) 降为 O(k * hops * avg_degree)

        Args:
            top_k_entities: 起点节点的 id 列表（来自实体链接结果）。
            db_id:          图谱标识符。
            max_hops:       最大 BFS 跳数（Cypher 变长路径上限）。
            max_triples:    返回三元组数量上限。

        Returns:
            格式化为 [Local Knowledge Subgraph] 块的字符串。
        """
        if self._driver is None:
            raise RuntimeError("Neo4j 未连接")
        if not top_k_entities:
            return ""

        cypher = f"""
        MATCH (seed:Entity {{db_id: $db_id}})
        WHERE seed.id IN $entity_ids
        MATCH (seed)-[r:RELATION*1..{max_hops}]->(neighbor:Entity {{db_id: $db_id}})
        WITH seed, r, neighbor
        LIMIT {max_triples}
        RETURN
            seed.label        AS src_label,
            seed.type         AS src_type,
            seed.description  AS src_desc,
            [rel IN r | rel.type][0] AS rel_type,
            neighbor.label    AS dst_label,
            neighbor.type     AS dst_type
        """

        try:
            with self._driver.session() as session:
                result = session.run(
                    cypher,
                    db_id=db_id,
                    entity_ids=top_k_entities,
                )
                records = list(result)
        except Exception as e:
            logger.warning(f"Cypher 子图查询失败：{e}")
            return ""

        if not records:
            return ""

        # 以源节点为核心聚合出边
        from collections import defaultdict
        out_edges: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
        src_meta: dict[str, tuple[str, str]] = {}

        for rec in records:
            src_label = rec["src_label"] or ""
            src_type  = rec["src_type"]  or ""
            src_desc  = rec["src_desc"]  or ""
            rel_type  = rec["rel_type"]  or "related_to"
            dst_label = rec["dst_label"] or ""
            dst_type  = rec["dst_type"]  or ""

            src_meta[src_label] = (src_type, src_desc)
            rel_readable = _RELATION_LABELS.get(rel_type, rel_type)
            out_edges[src_label].append((rel_readable, dst_label, dst_type, rel_type))

        lines = ["[Local Knowledge Subgraph]"]
        for src_label, edges_list in out_edges.items():
            src_type, src_desc = src_meta.get(src_label, ("", ""))
            lines.append(f"\n  [{src_type}: {src_label}]")
            if src_desc:
                lines.append(f"    Description: {src_desc[:120]}")
            for rel_readable, dst_label, dst_type, _ in edges_list[:8]:
                type_hint = f" ({dst_type})" if dst_type else ""
                lines.append(f"    → {rel_readable:<20}: {dst_label}{type_hint}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 辅助查询
    # ------------------------------------------------------------------

    def get_entity_neighbors(
        self,
        entity_id: str,
        db_id: str,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """
        查询单个实体的直接邻居。

        Args:
            entity_id: 节点 id。
            db_id:     图谱标识符。
            direction: "out" | "in" | "both"。

        Returns:
            [{"neighbor_id": ..., "neighbor_label": ..., "rel_type": ..., "weight": ...}]
        """
        if self._driver is None:
            raise RuntimeError("Neo4j 未连接")

        if direction == "out":
            match_clause = "MATCH (n:Entity {id: $eid, db_id: $db_id})-[r:RELATION]->(m:Entity {db_id: $db_id})"
        elif direction == "in":
            match_clause = "MATCH (n:Entity {id: $eid, db_id: $db_id})<-[r:RELATION]-(m:Entity {db_id: $db_id})"
        else:
            match_clause = "MATCH (n:Entity {id: $eid, db_id: $db_id})-[r:RELATION]-(m:Entity {db_id: $db_id})"

        cypher = f"""
        {match_clause}
        RETURN m.id AS neighbor_id, m.label AS neighbor_label,
               r.type AS rel_type, r.weight AS weight
        """
        try:
            with self._driver.session() as session:
                result = session.run(cypher, eid=entity_id, db_id=db_id)
                return [dict(rec) for rec in result]
        except Exception as e:
            logger.warning(f"邻居查询失败（entity_id={entity_id}）：{e}")
            return []
