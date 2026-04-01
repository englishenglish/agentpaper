"""PyMuPDF：降低 MuPDF 在终端刷屏（畸形 PDF 常见：no XObject subtype 等）。"""

_configured = False


def configure_fitz_mupdf_console() -> None:
    """
    关闭 MuPDF 通过 PyMuPDF「消息」通道输出的错误/警告显示。
    须在首次 fitz.open 之前调用效果最佳；可重复调用（幂等）。
    """
    global _configured
    if _configured:
        return
    try:
        import fitz

        tools = getattr(fitz, "TOOLS", None)
        if tools is not None:
            if hasattr(tools, "mupdf_display_errors"):
                tools.mupdf_display_errors(False)
            if hasattr(tools, "mupdf_display_warnings"):
                tools.mupdf_display_warnings(False)
    except Exception:
        pass
    _configured = True
