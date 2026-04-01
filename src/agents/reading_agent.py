import os
import re
import json
import ast
import asyncio
from typing import Any, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from pydantic import BaseModel, Field, field_validator

from src.utils.log_utils import setup_logger
from src.core.prompts import reading_agent_prompt, kg_extraction_prompt
from src.core.model_client import create_reading_model_client
from src.core.state_models import BackToFrontData, State, ExecutionState
from src.core.embedding import get_shared_embedder
from src.extraction.graph_builder import GraphBuilder, save_entity_graph, load_entity_graph
from src.extraction.community_builder import CommunityBuilder
from src.knowledge import knowledge_base

logger = setup_logger(__name__)

# 限制并发 LLM 调用数，避免触发 API 速率上限
_LLM_SEMAPHORE = asyncio.Semaphore(5)


class KeyMethodology(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description='方法名称（如 "Transformer-based Sentiment Classifier"）',
    )
    principle: Optional[str] = Field(default=None, description="核心原理")
    novelty: Optional[str] = Field(
        default=None,
        description='创新点（如「首次引入领域自适应预训练」）',
    )


class ExtractedPaperData(BaseModel):
    core_problem: str = Field(default=None, description="核心问题")
    key_methodology: KeyMethodology = Field(default=None, description="关键方法")
    datasets_used: List[str] = Field(default=[], description="使用的数据集")
    evaluation_metrics: List[str] = Field(default=[], description="评估指标")
    main_results: str = Field(default="", description="主要结果")
    limitations: str = Field(default="", description="局限性")
    contributions: List[str] = Field(default=[], description="贡献")

    @field_validator("datasets_used", "evaluation_metrics", "contributions", mode="before")
    @classmethod
    def _validate_list_fields(cls, v):
        """
        统一清洗列表字段，确保最终为 List[str]，并且不会出现 [None] 这种情况。
        支持以下输入形式：
        - None / ""           -> []
        - "SQuAD"             -> ["SQuAD"]
        - ["SQuAD", None, ""] -> ["SQuAD"]
        """
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple, set)):
            cleaned: list[str] = []
            for item in v:
                if item is None:
                    continue
                s = str(item).strip()
                if not s:
                    continue
                cleaned.append(s)
            return cleaned
        return [str(v)]

    @field_validator("core_problem", "main_results", "limitations", mode="before")
    @classmethod
    def _validate_str_fields(cls, v):
        if v is None:
            return ""
        return str(v)


class ReadingExtractedPapersData(BaseModel):
    """reading_agent 专用的论文数据列表，与 state_models.LegacyExtractedPapersData 分开定义"""
    papers: List[ExtractedPaperData] = Field(default=[], description="提取的论文数据列表")


_reading_model_client = None


def _get_reading_model_client():
    global _reading_model_client
    if _reading_model_client is None:
        _reading_model_client = create_reading_model_client()
    return _reading_model_client


def _make_read_agent() -> AssistantAgent:
    """每次调用创建新实例，避免多并发请求共享对话历史导致上下文污染"""
    return AssistantAgent(
        name="read_agent",
        model_client=_get_reading_model_client(),
        system_message=reading_agent_prompt,
        model_client_stream=True,
    )


async def _extract_kg_for_paper(paper: Dict[str, Any], extracted: Any) -> Dict[str, Any]:
    """
    调用 LLM（kg_extraction_prompt）从单篇论文抽取三元组。
    返回 {"entities": [...], "relations": [...]} 或空 dict（失败时）。
    """
    title = str(paper.get("title", ""))
    abstract = str(paper.get("abstract", paper.get("summary", "")))
    ext_dict: Dict[str, Any] = {}
    if extracted is not None:
        ext_dict = (
            extracted.model_dump() if hasattr(extracted, "model_dump") else
            extracted if isinstance(extracted, dict) else {}
        )

    text_input = (
        f"Title: {title}\n\n"
        f"Abstract: {abstract}\n\n"
        f"Methodology: {ext_dict.get('key_methodology', {})}\n"
        f"Datasets: {ext_dict.get('datasets_used', [])}\n"
        f"Metrics: {ext_dict.get('evaluation_metrics', [])}\n"
        f"Contributions: {ext_dict.get('contributions', [])}"
    )

    kg_agent = AssistantAgent(
        name="kg_extraction_agent",
        model_client=_get_reading_model_client(),
        system_message=kg_extraction_prompt,
        model_client_stream=False,
    )
    try:
        result = await kg_agent.run(task=text_input)
        raw = result.messages[-1].content
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
                return json.loads(repaired)
        if isinstance(raw, dict):
            return raw
    except Exception as e:
        logger.warning(f"KG 三元组抽取失败（{title[:40]}）: {e}")
    return {}


_get_embedder = get_shared_embedder


async def _enrich_graph_with_llm(
    builder: GraphBuilder,
    papers: List[Dict[str, Any]],
    extracted_papers: ReadingExtractedPapersData,
) -> None:
    """并行调用 LLM 为每篇论文抽取三元组，通过 GraphBuilder.merge_triples 合并进图谱"""

    async def _bounded_extract(paper, extracted):
        async with _LLM_SEMAPHORE:
            return await _extract_kg_for_paper(paper, extracted)

    tasks = [
        _bounded_extract(
            papers[i],
            extracted_papers.papers[i] if i < len(extracted_papers.papers) else None,
        )
        for i in range(len(papers))
    ]
    kg_outputs = await asyncio.gather(*tasks, return_exceptions=True)

    for i, kg_output in enumerate(kg_outputs):
        if isinstance(kg_output, Exception) or not isinstance(kg_output, dict):
            continue
        paper_id_raw = str(papers[i].get("paper_id") or papers[i].get("id") or f"paper_{i}")
        builder.merge_triples(kg_output, paper_id_raw=paper_id_raw)


async def _build_graph_for_papers(
    papers: List[Dict[str, Any]],
    extracted_papers: ReadingExtractedPapersData,
    db_id: str,
) -> None:
    """为论文构建知识图谱并增量持久化到指定永久知识库的 db_id。"""
    try:
        embedder = _get_embedder()
        existing_graph = load_entity_graph(db_id, embedder=embedder)

        builder = GraphBuilder(embedder=embedder, graph_data=existing_graph)
        builder.build_from_papers(papers, extracted_papers.papers, db_id)

        await _enrich_graph_with_llm(builder, papers, extracted_papers)

        CommunityBuilder(builder.graph).build_communities()

        save_entity_graph(db_id, builder.graph)
        logger.info(f"知识图谱构建完成：{builder.graph.get('stats', {})}")
    except Exception as e:
        logger.warning(f"实体图谱构建失败（不影响主流程）: {e}")


async def reading_node(state: State) -> State:
    """搜索论文节点"""
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.READING
    await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="initializing", data=None))

    papers = current_state.search_results

    # 当关闭联网搜索或未检索到新论文时，直接跳过阅读阶段
    if not papers:
        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.READING,
                state="completed",
                data="未检索到需要阅读的论文，将直接基于现有知识库（若有）进行问答。",
            )
        )
        return {"value": current_state}

    async def _run_one(paper: dict) -> Any:
        async with _LLM_SEMAPHORE:
            return await _make_read_agent().run(task=str(paper))

    results = await asyncio.gather(*[_run_one(p) for p in papers])

    extracted_papers = ReadingExtractedPapersData()
    successful_papers = []

    for i, result in enumerate(results):
        raw_content = result.messages[-1].content

        if isinstance(raw_content, ExtractedPaperData):
            extracted_papers.papers.append(raw_content)
            successful_papers.append(papers[i])
            continue
        if isinstance(raw_content, dict):
            data = raw_content
        elif isinstance(raw_content, str):
            clean_content = raw_content.strip()
            if clean_content.startswith("```"):
                clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content)
                clean_content = re.sub(r"\s*```$", "", clean_content)
            try:
                data = json.loads(clean_content)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(clean_content)
                except Exception:
                    logger.error(f"Failed to parse content as JSON or Python dict: {clean_content}")
                    continue
        else:
            logger.error(f"Unsupported content type: {type(raw_content)}")
            continue

        if isinstance(data, list):
            if len(data) > 0:
                data = data[0]
            else:
                logger.warning("Parsed content is an empty list.")
                continue

        if isinstance(data, dict):
            if "papers" in data and isinstance(data["papers"], list):
                if len(data["papers"]) > 0:
                    data = data["papers"][0]
            elif "paper" in data and isinstance(data["paper"], dict):
                data = data["paper"]

        try:
            parsed_paper = ExtractedPaperData.model_validate(data)
            extracted_papers.papers.append(parsed_paper)
            successful_papers.append(papers[i])
        except Exception as e:
            logger.error(f"Validation failed for data: {data}. Error: {e}")

    # 从会话级 state.config 获取选中的知识库（避免读全局 config 导致并发串库）
    cfg = getattr(current_state, "config", None) or {}
    sel = cfg.get("selected_db_ids") or []
    active_db_ids: List[str] = [sel[0]] if isinstance(sel, list) and sel else []

    if active_db_ids and successful_papers:
        await _build_graph_for_papers(successful_papers, extracted_papers, active_db_ids[0])
    else:
        logger.info("未选择永久知识库或无成功解析的论文，跳过知识图谱构建。")

    # 向量索引：将 PDF 写入本会话知识库，供后续 RAG 检索
    if active_db_ids and successful_papers:
        db_id = active_db_ids[0]
        for paper in successful_papers:
            path = paper.get("pdf_local_path")
            if path and os.path.isfile(path):
                try:
                    await knowledge_base.add_content(db_id, [path], params={"content_type": "file"})
                except Exception as e:
                    logger.warning(f"PDF 入库失败 {path}: {e}")

    current_state.extracted_data = extracted_papers
    await state_queue.put(BackToFrontData(
        step=ExecutionState.READING,
        state="completed",
        data=f"论文阅读完成，共阅读 {len(extracted_papers.papers)} 篇论文",
    ))
    return {"value": current_state}
