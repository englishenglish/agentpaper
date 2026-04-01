"""
预览「相邻句向量相似度」切块：PDF 可先走 MinerU Agent API 识别，再语义切分。

流程（推荐 PDF + 云端识别）::

    论文地址/本地 PDF
        → process_file_mineru_api（MinerU Agent，Markdown 正文）
        → embedding_sentence_chunk_chunks（相邻句向量相似度切块）

其它 OCR 仍走 ``indexing.process_file_to_markdown``（与主流程一致）。

用法（在项目根目录）::

    # 推荐：MinerU Agent API 识别 PDF → 再语义切块
    python test/test_embedding_chunk_preview.py "https://arxiv.org/pdf/xxxx.pdf" --enable-ocr mineru_agent_api

    python test/test_embedding_chunk_preview.py "D:/papers/foo.pdf" --enable-ocr mineru_agent_api --mineru-language en
    python test/test_embedding_chunk_preview.py "paper.pdf" --enable-ocr disable

依赖：``.env`` 与嵌入模型；``mineru_agent_api`` 需能访问 mineru.net。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _is_pdf_url(url: str) -> bool:
    p = (urlparse(url).path or "").lower()
    return p.endswith(".pdf") or "/pdf/" in p


def _download_pdf_to_temp(url: str) -> str:
    import requests

    r = requests.get(url, timeout=180, stream=True)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def _is_pdf_path(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


async def extract_pdf_text_with_mineru_agent_api(
    local_pdf_path: str,
    *,
    mineru_params: dict[str, Any] | None = None,
) -> str:
    """
    使用 ``OCRPlugin.process_file_mineru_api`` 识别 PDF（MinerU Agent 云端 API）。
    在测试脚本中显式展示「先识别、再切块」的第一步。
    """
    from src.ocr.ocr import OCRPlugin

    def _run() -> str:
        return OCRPlugin().process_file_mineru_api(local_pdf_path, params=mineru_params or {})

    return await asyncio.to_thread(_run)


async def _resolve_local_pdf_path(paper_ref: str) -> tuple[str, str]:
    """
    将「URL 或本地路径」解析为本地 PDF 路径。

    Returns:
        (本地绝对路径, 显示用文件名)
    """
    ref = paper_ref.strip()
    if ref.startswith("http://") or ref.startswith("https://"):
        if not _is_pdf_url(ref):
            raise ValueError("mineru_agent_api 仅支持 PDF 直链（含 /pdf/ 或 .pdf）")
        tmp = _download_pdf_to_temp(ref)
        name = Path(urlparse(ref).path).name or "paper.pdf"
        return tmp, name

    path = Path(ref).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"不是有效文件路径: {ref}")
    if not _is_pdf_path(path):
        raise ValueError("mineru_agent_api 仅用于 .pdf 文件")
    return str(path), path.name


async def load_text_for_preview(
    paper_ref: str,
    *,
    enable_ocr: str,
    mineru_params: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    得到用于切块的纯文本（及文件名）。

    - ``mineru_agent_api``：显式 ``process_file_mineru_api`` → 字符串（无 ``# 标题`` 包装）。
    - 其它：走 ``process_file_to_markdown``，与主工程一致。
    """
    from src.rag.indexing import process_file_to_markdown, process_url_to_markdown

    ref = paper_ref.strip()

    if enable_ocr == "mineru_agent_api":
        local_path, filename = await _resolve_local_pdf_path(ref)
        try:
            text = await extract_pdf_text_with_mineru_agent_api(
                local_path,
                mineru_params=mineru_params,
            )
            return (text or "").strip(), filename
        finally:
            if ref.startswith("http://") or ref.startswith("https://"):
                try:
                    os.unlink(local_path)
                except OSError:
                    pass

    if ref.startswith("http://") or ref.startswith("https://"):
        if _is_pdf_url(ref):
            tmp = _download_pdf_to_temp(ref)
            try:
                md = await process_file_to_markdown(
                    tmp,
                    params={"enable_ocr": enable_ocr},
                )
                name = Path(urlparse(ref).path).name or "paper.pdf"
                return md, name
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        md = await process_url_to_markdown(ref)
        return md, ref[:80]

    path = Path(ref)
    if not path.is_file():
        raise FileNotFoundError(f"不是有效文件路径: {ref}")
    md = await process_file_to_markdown(
        str(path.resolve()),
        params={"enable_ocr": enable_ocr},
    )
    return md, path.name


def _strip_leading_title_md(text: str) -> str:
    lines = text.split("\n")
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).lstrip("\n")
    return text


async def preview_embedding_chunks_for_paper(
    paper_ref: str,
    *,
    enable_ocr: str = "disable",
    strip_title: bool = True,
    chunk_params: dict[str, Any] | None = None,
    mineru_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    正文 → ``embedding_sentence_chunk_chunks`` → chunk 列表。

    当 ``enable_ocr == "mineru_agent_api"`` 时，正文来自 ``process_file_mineru_api``。
    """
    from src.rag.utils.embedding_sentence_chunk import embedding_sentence_chunk_chunks

    text, filename = await load_text_for_preview(
        paper_ref,
        enable_ocr=enable_ocr,
        mineru_params=mineru_params,
    )
    body = _strip_leading_title_md(text) if strip_title else text
    file_id = "preview_doc"
    return embedding_sentence_chunk_chunks(
        body,
        file_id,
        filename,
        params=chunk_params or {},
    )


def print_chunks(chunks: list[dict[str, Any]], preview_chars: int) -> None:
    print(f"\n共 {len(chunks)} 个 chunk\n")
    for i, c in enumerate(chunks):
        content = c.get("content", "")
        n = len(content)
        head = content[:preview_chars]
        if n > preview_chars:
            head += f"\n... （共 {n} 字符，已截断展示前 {preview_chars} 字符）"
        print(f"======== chunk {i} | chars={n} | id={c.get('id')} ========")
        print(head)
        print()


async def _async_main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF：可选 MinerU Agent API 识别 → embedding 语义切块预览",
    )
    parser.add_argument(
        "paper_ref",
        help="PDF 直链 / 本地 PDF；非 PDF 时勿用 mineru_agent_api",
    )
    parser.add_argument(
        "--enable-ocr",
        choices=["disable", "onnx_rapid_ocr", "mineru_ocr", "mineru_agent_api"],
        default="disable",
        help="PDF：disable=仅文本层；mineru_agent_api=先 process_file_mineru_api 识别再语义切块（推荐）",
    )
    parser.add_argument(
        "--mineru-language",
        default="en",
        help="仅 mineru_agent_api：language 参数（默认 en）",
    )
    parser.add_argument(
        "--mineru-page-range",
        default=None,
        help="仅 mineru_agent_api：如 1-10",
    )
    parser.add_argument(
        "--no-strip-title",
        action="store_true",
        help="保留 markdown 首行标题再切块（非 mineru_agent_api 时常见）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="相邻句余弦相似度阈值，默认读 config",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=None,
        help="单块最大字符数，默认读 config",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=600,
        help="终端每块展示字符数",
    )
    args = parser.parse_args()

    mineru_params: dict[str, Any] = {"language": args.mineru_language}
    if args.mineru_page_range:
        mineru_params["page_range"] = args.mineru_page_range

    chunk_params: dict[str, Any] = {}
    if args.threshold is not None:
        chunk_params["embedding_chunk_adjacent_threshold"] = args.threshold
    if args.max_chunk_chars is not None:
        chunk_params["embedding_chunk_max_chars"] = args.max_chunk_chars

    chunks = await preview_embedding_chunks_for_paper(
        args.paper_ref,
        enable_ocr=args.enable_ocr,
        strip_title=not args.no_strip_title,
        chunk_params=chunk_params or None,
        mineru_params=mineru_params,
    )
    print_chunks(chunks, args.preview_chars)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
