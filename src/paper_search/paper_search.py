import arxiv
import asyncio
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import time
import os
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class PaperSearcher:
    """论文搜索器，使用arxiv库搜索论文"""
    
    def __init__(self):
        """初始化论文搜索器"""
        pass
    
    async def search_papers(self, 
                      querys: List[str], 
                      max_results: int = 5,
                      sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance, 
                      sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending, 
                      start_date: Optional[Union[str, datetime]] = None, 
                      end_date: Optional[Union[str, datetime]] = None) -> List[Dict]:
        """
        搜索arXiv论文
        
        参数:
            querys: 搜索关键词
            max_results: 最大返回结果数量
            sort_by: 排序方式 (Relevance, LastUpdatedDate, SubmittedDate)
            sort_order: 排序顺序 (Ascending, Descending)
            start_date: 开始日期，可以是字符串(YYYY-MM-DD)或datetime对象
            end_date: 结束日期，可以是字符串(YYYY-MM-DD)或datetime对象
        
        返回:
            论文列表，每项包含论文的详细信息
        """
        # querys = ['artificial intelligence', 'AI', 'llm', 'machine learning', 'deep learning']
        try:
            # 构建搜索查询
            search_query = ""
            for query in querys:
                search_query += "all:%22"+query+"%22 OR "
            search_query = search_query[:-4]
            # 添加日期范围过滤
            if start_date or end_date:
                start_date_str = self._format_date(start_date) if start_date else "190001010000"
                end_date_str = self._format_date(end_date) if end_date else datetime.now().strftime("%Y%m%d2359")
                date_filter = f"submittedDate:[{start_date_str} TO {end_date_str}]"
                search_query = f"{search_query} AND {date_filter}"

            logger.info(f"开始搜索论文: query='{search_query}', max_results={max_results}, sort_by={sort_by}")


            logger.info(f"论文搜索查询条件: {search_query}")

            # 创建搜索对象
            try:
                search = arxiv.Search(
                    query=search_query,
                    max_results=max_results,
                    sort_by=sort_by,
                    sort_order=sort_order
                )
            except Exception as e:
                logger.error(f"创建arxiv搜索对象失败: {str(e)}")
                return []
            
            # logger.info(f"论文搜索结果为：{search.results()}")
            # 执行搜索并解析结果
            # 使用新方法格式化论文列表
            papers = []
            max_retries = 3  # 最大重试次数
            
            for attempt in range(max_retries):
                try:
                    # 只有在这里的 list() 操作时，才会真正向 arXiv 发起 HTTP 请求
                    papers = self.format_papers_list(search.results())
                    break  # 如果成功拿到数据，就跳出循环
                    
                except Exception as e:
                    error_msg = str(e)
                    # 捕获 429 限流错误或空页面错误
                    if "429" in error_msg or "Too Many Requests" in error_msg or "UnexpectedEmptyPageError" in error_msg:
                        wait_time = (attempt + 1) * 5  # 第一轮等 5 秒，第二轮等 10 秒...
                        logger.warning(f"⚠️ 触发 arXiv 限流 (HTTP 429)。程序休眠 {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        
                        if attempt == max_retries - 1:
                            logger.error("❌ 达到最大重试次数，arXiv 搜索失败。请稍后重试或减少搜索并发量。")
                            return []
                    else:
                        # 如果是其他网络断开等严重错误，直接抛出
                        logger.error(f"论文搜索数据拉取失败: {error_msg}")
                        raise e
            # =============== 修改的部分到这里结束 ===============

            logger.info(f"论文搜索完成，共找到 {len(papers)} 篇论文")
            return papers
        except Exception as e:
            logger.error(f"论文搜索失败: {str(e)}")
            raise
    
    async def search_by_topic(self, 
                       topic: str, 
                       limit: int = 10, 
                       recent_days: Optional[int] = None) -> List[Dict]:
        """
        按主题搜索最近的论文
        
        参数:
            topic: 主题关键词
            limit: 返回结果数量限制
            recent_days: 搜索最近多少天的论文，None表示不限制
        
        返回:
            论文列表
        """
        logger.info(f"按主题搜索论文: topic='{topic}', limit={limit}, recent_days={recent_days}")
        
        # 计算开始日期
        start_date = None
        if recent_days:
            start_date = datetime.now() - timedelta(days=recent_days)
        
        # 调用搜索方法
        return self.search_papers(
            query=topic,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
            start_date=start_date
        )

    async def download_and_extract(
        self,
        paper: Dict,
        dirpath: str,
    ) -> Tuple[Optional[str], str]:
        """下载论文 PDF 并提取正文。

        顺序：OCRPlugin.process_file_mineru_api → process_file_mineru_token → PyMuPDF 文本层。
        返回 (本地PDF路径, 提取的全文)。下载失败时返回 (None, "")。
        """
        paper_id = paper.get("paper_id", "")
        title = str(paper.get("title", paper_id))
        safe_title = "".join(c for c in title[:60] if c.isalnum() or c in " _-").strip()
        filename = f"{paper_id.replace('/', '_')}_{safe_title}.pdf"

        os.makedirs(dirpath, exist_ok=True)
        pdf_path = os.path.join(dirpath, filename)

        if not os.path.exists(pdf_path):
            try:

                def _do_download():
                    search = arxiv.Search(id_list=[paper_id])
                    result = next(search.results())
                    result.download_pdf(dirpath=dirpath, filename=filename)

                await asyncio.to_thread(_do_download)
                logger.info(f"PDF 下载成功: {pdf_path}")
            except Exception as e:
                logger.warning(f"PDF 下载失败 [{paper_id}]: {e}")
                return None, ""

        if not os.path.exists(pdf_path):
            return None, ""

        full_text = ""

        try:
            from src.ocr.ocr import OCRPlugin

            # --- 1) MinerU Agent 轻量解析 API ---
            try:

                def _extract_mineru_api():
                    return OCRPlugin().process_file_mineru_api(pdf_path)

                ocr_text = await asyncio.to_thread(_extract_mineru_api)
                full_text = (ocr_text or "").strip()
                if full_text:
                    logger.info(
                        f"MinerU Agent API 提取成功 [{paper_id}]: {len(full_text)} 字符"
                    )
            except Exception as e:
                logger.warning(
                    f"MinerU Agent API 失败 [{paper_id}]: {e}，尝试 MinerU Token API"
                )

            # --- 2) MinerU V4 Token API ---
            if not full_text:
                try:

                    def _extract_mineru_token():
                        return OCRPlugin().process_file_mineru_token(pdf_path)

                    ocr_text = await asyncio.to_thread(_extract_mineru_token)
                    full_text = (ocr_text or "").strip()
                    if full_text:
                        logger.info(
                            f"MinerU Token API 提取成功 [{paper_id}]: {len(full_text)} 字符"
                        )
                    else:
                        logger.warning(
                            f"MinerU Token 返回空文本 [{paper_id}]，尝试 PyMuPDF"
                        )
                except Exception as e:
                    logger.warning(
                        f"MinerU Token 提取异常 [{paper_id}]: {e}，尝试 PyMuPDF"
                    )

        except Exception as e:
            logger.warning(f"OCR 模块不可用或初始化失败 [{paper_id}]: {e}")

        # --- 3) PyMuPDF 文本层（数字版 PDF 效果好）---
        if not full_text:
            try:
                import fitz

                def _extract_fitz():
                    doc = fitz.open(pdf_path)
                    try:
                        return "\n".join(page.get_text() for page in doc)
                    finally:
                        doc.close()

                full_text = (await asyncio.to_thread(_extract_fitz)).strip()
                if full_text:
                    logger.info(f"PyMuPDF 文本提取成功 [{paper_id}]: {len(full_text)} 字符")
                else:
                    logger.warning(f"PyMuPDF 也未提取到文本 [{paper_id}]，将使用摘要")
            except Exception as e2:
                logger.warning(f"PyMuPDF 提取失败 [{paper_id}]: {e2}")

        if not full_text:
            logger.warning(f"未提取到 PDF 正文 [{paper_id}]，调用方可使用摘要等降级方案")

        return pdf_path, full_text

    async def download_paper(self, paper_id: str, dirpath: str = "./papers", filename: Optional[str] = None) -> Optional[str]:
        """
        根据论文ID下载PDF文件

        参数:
            paper_id: 论文的短ID (例如 '2310.06825')
            dirpath: 保存路径，默认为当前目录下的 papers 文件夹
            filename: 自定义文件名，如果为None则使用默认的 "ID.pdf" 或 "IDvX.pdf"

        返回:
            下载后的文件绝对路径，如果失败则返回 None
        """
        # 确保保存目录存在
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            logger.info(f"创建论文保存目录: {dirpath}")

        try:
            # 通过 ID 准确查找这篇论文
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())

            logger.info(f"正在下载论文: {paper.title} ({paper_id})...")

            # 调用 arxiv 库自带的下载方法
            # 可以在这里使用自定义文件名，比如 f"{paper_id}_{paper.title}.pdf"
            file_path = paper.download_pdf(dirpath=dirpath, filename=filename)

            logger.info(f"✅ 论文下载成功，已保存至: {file_path}")
            return file_path

        except StopIteration:
            logger.error(f"❌ 找不到 ID 为 {paper_id} 的论文")
            return None
        except Exception as e:
            logger.error(f"❌ 下载论文失败: {str(e)}")
            return None
    
    def format_papers_list(self, search_results) -> List[Dict]:
        """
        将搜索结果（迭代器或列表）格式化为论文信息字典列表
        
        参数:
            search_results: arxiv搜索结果对象（可能是迭代器）
        
        返回:
            格式化后的论文信息字典列表
        """
        # 将迭代器转换为列表以便后续处理
        results_list = list(search_results)
        
        # 格式化论文列表
        formatted_papers = [self._parse_paper_result(result) for result in results_list]
        
        logger.info(f"开始格式化论文列表，共 {len(results_list)} 篇论文")
        return formatted_papers

    def search_by_author(self, 
                        author_name: str, 
                        limit: int = 10) -> List[Dict]:
        """
        按作者搜索论文
        
        参数:
            author_name: 作者姓名
            limit: 返回结果数量限制
        
        返回:
            论文列表
        """
        logger.info(f"按作者搜索论文: author='{author_name}', limit={limit}")
        
        # 使用作者字段搜索
        query = f"au:{author_name}"
        return self.search_papers(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
    
    def _parse_paper_result(self, result: arxiv.Result) -> Dict:
        """
        解析arXiv搜索结果
        
        参数:
            result: arxiv.Result对象
        
        返回:
            包含论文信息的字典
        """
        # 从结果URL中提取论文ID
        paper_id = result.get_short_id()
        
        # 提取发布年份
        published_year = result.published.year if result.published else None
        
        return {
            "paper_id": paper_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": published_year,
            "published_date": result.published.isoformat() if result.published else None,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "doi": result.doi if hasattr(result, 'doi') else None
        }
    
    def _format_date(self, date: Union[str, datetime]) -> str:
        """
        
        格式化日期为arXiv API支持的格式YYYYMMDDTTTT
        
        参数:
            date: 日期字符串或datetime对象
        
        返回:
            格式化后的日期字符串YYYYMMDD0000
        """
        if isinstance(date, datetime):
            return date.strftime("%Y%m%d0000")
        elif isinstance(date, str):
            # 定义多种可能的日期格式
            date_formats = [
                "%Y-%m-%d",      # YYYY-MM-DD
                "%Y/%m/%d",      # YYYY/MM/DD
                "%Y.%m.%d",      # YYYY.MM.DD
                "%Y-%m",         # YYYY-MM
                "%Y/%m",         # YYYY/MM
                "%Y",            # YYYY
                "%Y年%m月%d日",  # 中文格式
                "%Y年%m月",      # 中文格式（年月）
                "%Y年",          # 中文格式（年）
            ]
            
            for fmt in date_formats:
                try:
                    if fmt == "%Y":  # 单独处理只有年份的情况
                        if len(date) == 4 and date.isdigit():
                            parsed_date = datetime(int(date), 1, 1)
                            return parsed_date.strftime("%Y%m%d0000")
                    elif fmt in ["%Y-%m", "%Y/%m", "%Y年%m月"]:  # 处理年月格式
                        try:
                            parsed_date = datetime.strptime(date, fmt)
                            return parsed_date.strftime("%Y%m%d0000")
                        except ValueError:
                            continue
                    else:
                        parsed_date = datetime.strptime(date, fmt)
                        return parsed_date.strftime("%Y%m%d0000")
                except ValueError:
                    continue
            
            # 如果所有格式都失败，尝试使用dateutil或返回默认值
            try:
                from dateutil import parser
                parsed_date = parser.parse(date)
                return parsed_date.strftime("%Y%m%d0000")
            except Exception:
                # 最终fallback：当前日期
                return datetime.now().strftime("%Y%m%d0000")
        
        # 默认返回当前日期
        return datetime.now().strftime("%Y%m%d0000")

# 示例用法
# 示例用法
if __name__ == "__main__":
    import asyncio


    async def main():
        searcher = PaperSearcher()

        # 1. 搜索 LLM 论文
        print("开始搜索...")
        papers = await searcher.search_papers(querys=["Retrieval-Augmented Generation", "LLM"], max_results=2)

        for p in papers:
            print(f"找到论文: {p['title']} | ID: {p['paper_id']}")

        # 2. 如果找到了论文，尝试下载第一篇
        if papers:
            first_paper_id = papers[0]['paper_id']
            print(f"\n准备下载第一篇论文: {first_paper_id}")

            # 调用新增的下载方法
            downloaded_path = await searcher.download_paper(
                paper_id=first_paper_id,
                dirpath="./my_downloaded_papers"
            )

            if downloaded_path:
                print(f"下载完成！文件位置在: {downloaded_path}")


    # 运行异步函数
    asyncio.run(main())