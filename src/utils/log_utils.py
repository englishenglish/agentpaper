import logging
import sys
from pathlib import Path


def quiet_noisy_dependency_loggers():
    """
    抑制 AutoGen / OpenAI 客户端等依赖在 INFO 下打印的整段 JSON（LLMCall、LLMStreamStart/End 等）。
    不影响应用内 setup_logger 命名的 logger。
    """
    for name in (
        "autogen_agentchat",
        "autogen_ext",
        "autogen_core",
        "openai",
        "httpx",
        "httpcore",
        "httpcore.http11",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _ensure_utf8_stdio():
    """Windows 控制台默认编码常为 cp936，易导致中文日志乱码；尽量改为 UTF-8。"""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def setup_logger(name='project', log_file='project.log', level=logging.DEBUG):
    """设置日志记录器。"""
    # 创建日志目录（如果不存在）
    log_dir = Path("output/log")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 确保日志文件保存到log_dir目录下
    log_file_path = log_dir / log_file  # 组合目录和文件名

    _ensure_utf8_stdio()

    # 创建日志记录器，避免重复添加处理器
    logger = logging.getLogger(name)
    if logger.handlers:  # 防止重复配置
        return logger
    logger.setLevel(level)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 尽早压低第三方 INFO，避免在 import 顺序下先于 setup_logger 加载 autogen 时刷屏
quiet_noisy_dependency_loggers()


if __name__ == '__main__':
    logger = setup_logger()
    logger.info('This is an info message')
    logger.error('This is an error message')