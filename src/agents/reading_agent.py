import sys
import os
import re
import json
import ast
import asyncio
import shutil
from typing import List, Optional, Dict, Any

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autogen_agentchat.agents import AssistantAgent
from pydantic import BaseModel, Field, field_validator
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.log_utils import setup_logger
from src.utils import hashstr
from src.utils.datetime_utils import utc_isoformat
from src.core.prompts import reading_agent_prompt, kg_extraction_prompt
from src.core.model_client import create_default_client, create_reading_model_client
from src.core.state_models import BackToFrontData, State, ExecutionState
from src.services.chroma_client import ChromaClient
from src.services.graph_store import (
    build_entity_graph_from_papers,
    merge_kg_triples,
    build_communities,
    save_entity_graph,
    load_entity_graph,
)
from src.knowledge.knowledge import knowledge_base
from src.core.config import config

logger = setup_logger(__name__)


class KeyMethodology(BaseModel):
    name: Optional[str] = Field(default=None, description="方法名称（如“Transformer-based Sentiment Classifier”）")
    principle: Optional[str] = Field(default=None, description="核心原理")
    novelty: Optional[str] = Field(default=None, description="创新点（如“首次引入领域自适应预训练”）")


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
        # 单个字符串 -> [str]
        if isinstance(v, str):
            return [v]
        # 列表/可迭代 -> 过滤 None 和空串，统一转 str
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
        # 其他类型，兜底转为单元素字符串列表
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


model_client = create_reading_model_client()


def _make_read_agent() -> AssistantAgent:
    """每次调用创建新实例，避免多并发请求共享对话历史导致上下文污染"""
    return AssistantAgent(
        name="read_agent",
        model_client=model_client,
        system_message=reading_agent_prompt,
        model_client_stream=True,
    )


def sanitize_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    """清洗元数据，确保 ChromaDB 能够接受（只能是 str, int, float, bool）"""
    new_meta = {}
    for k, v in paper.items():
        if v is None:
            continue
        if isinstance(v, list):
            new_meta[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            new_meta[k] = json.dumps(v, ensure_ascii=False)
        else:
            new_meta[k] = str(v)  # 强制转为字符串以防特殊类型报错
    return new_meta


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
        model_client=model_client,
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
                # LLM 偶发输出非法转义（如 \_ / \( / \-），先修复后再解析
                repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
                return json.loads(repaired)
        if isinstance(raw, dict):
            return raw
    except Exception as e:
        logger.warning(f"KG 三元组抽取失败（{title[:40]}）: {e}")
    return {}


async def _enrich_graph_with_llm(
    graph: Dict[str, Any],
    papers: List[Dict[str, Any]],
    extracted_papers: ReadingExtractedPapersData,
) -> None:
    """并行调用 LLM 为每篇论文抽取三元组，合并进图谱"""
    tasks = [
        _extract_kg_for_paper(
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
        merge_kg_triples(graph, kg_output, paper_id_raw=paper_id_raw)


async def add_papers_to_kb(
    papers: Optional[List[Dict[str, Any]]],
    extracted_papers: ReadingExtractedPapersData,
    kb_name: str,
    kb_description: str,
) -> None:
    """【RAG 改造核心】将提取的论文文本进行切片（Chunking），并结合结构化标签存入知识库。

    kb_name / kb_description 由上游根据当前对话主题 & 论文内容动态生成，
    避免所有知识库都叫“临时问答知识库”。
    """
    if not papers:
        logger.warning("No papers to add to KB.")
        return

    embedding_dic = config.get("embedding-model")
    embedding_provider = embedding_dic.get("model-provider")
    provider_dic = config.get(embedding_provider)

    embed_info = {
        "name": embedding_dic.get("model"),
        "dimension": embedding_dic.get("dimension"),
        "base_url": provider_dic.get("base_url"),
        "api_key": provider_dic.get("api_key"),
    }
    kb_type = config.get("KB_TYPE")

    # 创建专为本次问答准备的知识库（名称/描述由对话 & 论文语义生成）
    database_info = await knowledge_base.create_database(
        kb_name,
        kb_description,
        kb_type=kb_type,
        embed_info=embed_info,
        llm_info=None,
    )
    db_id = database_info["db_id"]
    config.set("tmp_db_id", db_id)

    # KB uploads 目录：已下载的临时 PDF 将被复制到此处，作为知识库的持久化文件
    save_dir = config.get("SAVE_DIR", "data")
    kb_uploads_dir = os.path.join(
        save_dir, "knowledge_base_data", "chroma_data", f"kb_{db_id}", "uploads"
    )
    os.makedirs(kb_uploads_dir, exist_ok=True)

    # 1. 初始化文本切分器（按 800 字符切块，保留 100 字符重叠防止上下文截断）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    documents = []
    metadatas = []
    ids = []
    file_records: list[dict] = []

    # 2. 遍历论文并进行切片
    for i, (paper, extracted_paper) in enumerate(zip(papers, extracted_papers.papers)):
        # 获取需要切分的文本：优先取全文 full_text，如果没有则取 summary 或 abstract
        content_to_chunk = paper.get('full_text', paper.get('summary', paper.get('abstract', '')))

        # 兜底：如果既没有全文也没有摘要，降级使用大模型提取出来的 JSON 字符串作为文本
        if not content_to_chunk or not str(content_to_chunk).strip():
            content_to_chunk = json.dumps(extracted_paper.model_dump(), ensure_ascii=False)

        # 切片
        chunks = text_splitter.split_text(str(content_to_chunk))

        # 生成该论文的稳定 file_id（基于 paper_id/title + db_id，确保可重复、不冲突）
        paper_id_raw = str(paper.get("paper_id") or paper.get("id") or paper.get("title") or f"paper_{i}")
        file_id = f"arxiv_{hashstr(paper_id_raw + db_id, 12)}"

        # 准备元数据：合并基础 metadata 和 大模型提取出来的核心字段（利于混合检索过滤）
        base_meta = sanitize_metadata(paper)
        extracted_dict = extracted_paper.model_dump()

        rich_meta_base = {
            **base_meta,
            "core_problem": str(extracted_dict.get("core_problem", "")),
            "methodology_name": str(extracted_dict.get("key_methodology", {}).get("name", "")),
            "paper_index": i,
            "full_doc_id": file_id,   # 供 get_file_content 按文档过滤
            "source": str(paper.get("title") or paper_id_raw),
        }

        # 将每个 chunk 加入入库列表（每 chunk 独立携带 chunk_index 以支持排序）
        for j, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({**rich_meta_base, "chunk_index": j})
            ids.append(f"{file_id}_chunk_{j}")

        # 将临时 PDF 复制到 KB uploads 目录，并将路径更新为 KB 内的持久路径
        title = str(paper.get("title") or paper_id_raw)
        safe_title = "".join(c for c in title[:60] if c.isalnum() or c in " _-").strip()
        dest_filename = f"{paper_id_raw.replace('/', '_')}_{safe_title}.pdf"
        dest_path = os.path.join(kb_uploads_dir, dest_filename)

        tmp_pdf_path = paper.get("pdf_local_path")
        if tmp_pdf_path and os.path.exists(tmp_pdf_path) and not os.path.exists(dest_path):
            try:
                shutil.copy2(tmp_pdf_path, dest_path)
                logger.info(f"PDF 已复制到知识库目录: {dest_path}")
            except Exception as e:
                logger.warning(f"复制 PDF 失败，使用原始路径: {e}")
                dest_path = tmp_pdf_path

        # 构建该论文的 files_meta 注册记录
        file_records.append({
            "file_id": file_id,
            "database_id": db_id,
            "filename": f"{safe_title}.pdf" if safe_title else f"{paper_id_raw}.pdf",
            "path": dest_path if os.path.exists(dest_path) else str(paper.get("pdf_url") or paper.get("url") or paper_id_raw),
            "file_type": "arxiv",
            "status": "done",
            "created_at": utc_isoformat(),
            "source_type": "arxiv",
            "paper_id": paper_id_raw,
            "authors": str(paper.get("authors") or ""),
            "abstract": str(paper.get("abstract") or paper.get("summary") or "")[:500],
        })

    if not documents:
        logger.warning("No documents generated after chunking.")
        return

    data = {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
    }

    # 执行最终的数据入库
    await knowledge_base.add_processed_content(db_id, data)

    # 将每篇论文注册为独立的文档记录，使其出现在前端文档列表中
    try:
        knowledge_base.register_file_records(db_id, file_records)
        logger.info(f"已注册 {len(file_records)} 篇 arXiv 论文到知识库文档列表（db_id={db_id}）")
    except Exception as e:
        logger.warning(f"注册 arXiv 论文文档记录失败（不影响检索功能）: {e}")

    # ---- 知识图谱构建（两阶段）----
    try:
        # 阶段 1：从结构化字段构建初始图谱（快速，无 LLM 调用）
        existing_graph = load_entity_graph(db_id)
        if existing_graph:
            # 增量更新：将新论文实体合并进已有图谱
            new_graph = build_entity_graph_from_papers(papers, extracted_papers.papers, db_id)
            # 合并节点和边（边去重，避免重复入库时产生重复边）
            for nid, node in new_graph["nodes"].items():
                if nid not in existing_graph["nodes"]:
                    existing_graph["nodes"][nid] = node
            existing_edge_keys = {
                (e["source"], e["target"], e["type"]) for e in existing_graph["edges"]
            }
            for edge in new_graph["edges"]:
                key = (edge["source"], edge["target"], edge["type"])
                if key not in existing_edge_keys:
                    existing_graph["edges"].append(edge)
                    existing_edge_keys.add(key)
            for pid, ents in new_graph["paper_entities"].items():
                existing_graph["paper_entities"].setdefault(pid, [])
                for e in ents:
                    if e not in existing_graph["paper_entities"][pid]:
                        existing_graph["paper_entities"][pid].append(e)
            existing_graph["entity_aliases"].update(new_graph["entity_aliases"])
            graph_payload = existing_graph
        else:
            graph_payload = build_entity_graph_from_papers(papers, extracted_papers.papers, db_id)

        # 阶段 2：调用 LLM 抽取精细三元组并合并（每篇论文独立调用，并行执行）
        await _enrich_graph_with_llm(graph_payload, papers, extracted_papers)

        # 阶段 3：Louvain 社区聚类
        build_communities(graph_payload)

        # 持久化
        save_entity_graph(db_id, graph_payload)
        logger.info(f"知识图谱构建完成：{graph_payload['stats']}")
    except Exception as e:
        logger.warning(f"实体图谱构建失败（不影响主流程）: {e}")


async def reading_node(state: State) -> State:
    """搜索论文节点"""
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.READING
    await state_queue.put(BackToFrontData(step=ExecutionState.READING,state="initializing",data=None))

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

    # 每篇论文创建独立的 Agent 实例并行执行，避免共享对话历史导致上下文污染
    results = await asyncio.gather(*[_make_read_agent().run(task=str(paper)) for paper in papers])

    # 合并结果
    extracted_papers = ReadingExtractedPapersData()
    # 注释掉原本的代码，防止数据格式导致报错
    # for result in results:
    #     if result.messages[-1].content:
    #         parsed_paper = result.messages[-1].content
    #         extracted_papers.papers.append(parsed_paper)   
    
    # 清洗和预处理获取的数据    
    successful_papers = []
    for i, result in enumerate(results):
        raw_content = result.messages[-1].content
        # logger.info(f"Reading Agent Raw Output: {raw_content}") # 打印原始输出
        
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

        # 清理 Markdown 代码块
        # 3. 数据结构修正（处理列表包裹或 {"papers": ...} 包裹）
        if isinstance(data, list):
            if len(data) > 0:
                data = data[0] # 取第一个
            else:
                logger.warning("Parsed content is an empty list.")
                continue
        
        if isinstance(data, dict):
            # 如果被包裹在 "papers" 键中
            if "papers" in data and isinstance(data["papers"], list):
                if len(data["papers"]) > 0:
                    data = data["papers"][0]
            # 如果被包裹在 "paper" 键中
            elif "paper" in data and isinstance(data["paper"], dict):
                data = data["paper"]
        
        try:
            # 4. 验证并转换
            parsed_paper = ExtractedPaperData.model_validate(data)
            extracted_papers.papers.append(parsed_paper)
            successful_papers.append(papers[i])
        except Exception as e:
            logger.error(f"Validation failed for data: {data}. Error: {e}")

    # 为本次问答动态生成知识库名称与描述，避免统一使用“临时问答知识库”
    topic = (current_state.current_question or current_state.user_request or "").strip()
    if topic:
        short_topic = topic[:40]
        kb_name = f"问答知识库：{short_topic}"
    else:
        kb_name = "问答知识库"

    paper_titles = []
    for p in successful_papers[:3]:
        title = str(p.get("title") or p.get("paper_title") or "").strip()
        if title:
            paper_titles.append(title)
    paper_part = f"，主要涵盖：{'；'.join(paper_titles)}" if paper_titles else ""
    kb_description = (
        f"围绕用户问题「{topic or '未提供主题'}」自动构建的问答知识库，"
        f"共收录 {len(successful_papers)} 篇相关论文的语义切片{paper_part}。"
    )

    await add_papers_to_kb(successful_papers, extracted_papers, kb_name, kb_description)

    current_state.extracted_data = extracted_papers
    await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="completed", data=f"论文阅读完成，共阅读 {len(extracted_papers.papers)} 篇论文"))
    return {"value": current_state}


if __name__ == "__main__":
    paper = {
        'core_problem': 'Despite the rapid introduction of autonomous vehicles, public misunderstanding and mistrust are prominent issues hindering their acceptance.'
    }
    chroma_client = ChromaClient()
    chroma_client.add_documents(
        documents=[paper],
        metadatas=[paper],
    )   
