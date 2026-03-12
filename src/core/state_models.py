from asyncio import Queue
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from enum import Enum


class BackToFrontData(BaseModel):
    step: str
    state: str
    data: Any


class ExecutionState(str, Enum):
    """工作流执行状态枚举"""
    INITIALIZING = "initializing"
    SEARCHING = "searching"
    READING = "reading"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    WRITING_DIRECTOR = "writing_director"
    SECTION_WRITING = "section_writing"
    WRITING = "writing"
    REPORTING = "reporting"

    # RAG 问答专属状态
    QA_ANSWERING = "qa_answering"

    COMPLETED = "completed"
    FAILED = "failed"
    FINISHED = "finished"


class KeyMethodology(BaseModel):
    name: str
    principle: str
    novelty: str


class LegacyExtractedPaperData(BaseModel):
    """旧版论文抽取数据模型（用于长文生成流程）。
    RAG 问答流程使用 reading_agent.py 中的 ExtractedPaperData（含 validator 和默认值）。
    两者分开定义以避免命名冲突。
    """
    paper_id: str
    core_problem: str
    key_methodology: KeyMethodology
    datasets_used: List[str]
    evaluation_metrics: List[str]
    main_results: str
    limitations: str
    contributions: List[str]


class LegacyExtractedPapersData(BaseModel):
    """旧版论文抽取数据列表（用于长文生成流程）"""
    papers: List[LegacyExtractedPaperData]


# 向后兼容别名，避免破坏可能存在的外部引用
ExtractedPaperData = LegacyExtractedPaperData
ExtractedPapersData = LegacyExtractedPapersData


class AnalysisResults(BaseModel):
    """分析模块产生的结构化结果"""
    topic_clusters: Optional[Dict[str, List[str]]] = Field(default=None,
                                                           description="主题聚类, key: 主题名, value: 相关paper_id列表")
    trend_analysis: Optional[Dict[int, int]] = Field(default=None, description="趋势分析, key: 年份, value: 论文数量")
    method_comparison: Optional[List[Dict[str, Any]]] = Field(default=None, description="方法对比表格数据")
    influential_authors: Optional[List[str]] = Field(default=None, description="高产作者列表")
    influential_institutions: Optional[List[str]] = Field(default=None, description="核心机构列表")


class NodeError(BaseModel):
    search_node_error: Optional[str] = Field(default=None, description="搜索节点错误信息")
    reading_node_error: Optional[str] = Field(default=None, description="阅读节点错误信息")
    analyse_node_error: Optional[str] = Field(default=None, description="分析节点错误信息")
    writing_node_error: Optional[str] = Field(default=None, description="写作节点错误信息")
    report_node_error: Optional[str] = Field(default=None, description="报告生成节点错误信息")
    qa_node_error: Optional[str] = Field(default=None, description="问答节点错误信息")
    error: Optional[str] = Field(default=None, description="错误信息")


class PaperAgentState(BaseModel):
    """LangGraph工作流的全局状态对象"""
    frontend_data: Optional[BackToFrontData] = Field(default=None, description="前端展示数据")
    agent_logs: Dict[str, str] = Field(default_factory=dict, description="各智能体执行日志，key为智能体名称")

    user_request: str = Field(description="用户的原始输入请求/搜索主题")
    max_papers: int = Field(default=50, description="最大论文数量")

    # RAG 问答专用数据流字段
    current_question: Optional[str] = Field(default=None, description="当前用户提出的具体问题")
    qa_answer: Optional[str] = Field(default=None, description="大模型基于知识库生成的最新回答")
    chat_history: List[Dict[str, str]] = Field(default_factory=list,
                                               description="多轮对话历史，格式如 [{'role':'user', 'content':'...'}, ...]")
    retrieved_contexts: List[str] = Field(default_factory=list,
                                          description="本次问答检索到的所有文献片段(用于前端展示引用来源)")

    # 执行状态
    current_step: ExecutionState = Field(default=ExecutionState.INITIALIZING, description="当前执行步骤")
    error: NodeError = Field(default_factory=NodeError, description="错误信息")

    # 数据流（保留原有长文生成的字段，确保系统向前兼容）
    # extracted_data 实际存入的是 reading_agent.ExtractedPapersData 实例，
    # 使用 Any 避免与 LegacyExtractedPapersData 类型冲突
    search_results: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="检索到的论文元数据列表")
    paper_contents: Optional[Dict[str, str]] = Field(default_factory=dict,
                                                     description="解析后的论文全文字典, key: paper_id, value: 文本内容")
    extracted_data: Optional[Any] = Field(default=None, description="提取后的结构化信息（reading_agent.ExtractedPapersData）")
    analyse_results: Optional[str] = Field(default=None, description="分析洞察结果")
    outline: Optional[str] = Field(default=None, description="报告大纲")
    writted_sections: Optional[List[str]] = Field(default=None, description="已写章节内容")
    report_markdown: Optional[str] = Field(default=None, description="最终生成的Markdown报告内容")

    # 配置与上下文
    llm_provider: Any = Field(default=None, description="LLM提供者实例", exclude=True)
    config: Dict[str, Any] = Field(default_factory=dict, description="运行时配置")


class State(TypedDict):
    """LangGraph兼容的状态定义"""
    state_queue: Queue
    value: PaperAgentState


class ConfigSchema(TypedDict):
    """LangGraph兼容的配置定义"""
    state_queue: Queue
    value: Dict[str, Any]
