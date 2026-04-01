import os
import time
import uuid
from collections import defaultdict

import fitz  # fitz 即 pip install PyMuPDF
import numpy as np  # Added import for numpy
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm
from src.parsers.mineru import MinerUV4Client, MinerUV4Config
from src.utils import logger
from src.core.config import config

GOLBAL_STATE = {}

# OCR 服务监控统计
OCR_STATS = {"requests": defaultdict(int), "failures": defaultdict(int), "service_status": defaultdict(str)}


def log_ocr_request(service_name: str, file_path: str, success: bool, processing_time: float, error_msg: str = None):
    """记录 OCR 请求统计信息"""
    # 更新统计
    OCR_STATS["requests"][service_name] += 1

    if not success:
        OCR_STATS["failures"][service_name] += 1
        OCR_STATS["service_status"][service_name] = "error"
        logger.error(f"OCR失败 - {service_name}: {os.path.basename(file_path)} - {error_msg}")
    else:
        OCR_STATS["service_status"][service_name] = "healthy"
        logger.info(f"OCR成功 - {service_name}: {os.path.basename(file_path)}")


def get_ocr_stats():
    """获取 OCR 服务统计信息"""
    stats = {}
    for service in OCR_STATS["requests"]:
        success_count = OCR_STATS["requests"][service] - OCR_STATS["failures"][service]
        success_rate = (success_count / OCR_STATS["requests"][service]) if OCR_STATS["requests"][service] > 0 else 0

        stats[service] = {
            "total_requests": OCR_STATS["requests"][service],
            "success_count": success_count,
            "failure_count": OCR_STATS["failures"][service],
            "success_rate": f"{success_rate:.2%}",
            "status": OCR_STATS["service_status"][service],
        }

    return stats


class OCRServiceException(Exception):
    """OCR 服务异常"""

    def __init__(self, message, service_name=None, status_code=None):
        super().__init__(message)
        self.service_name = service_name
        self.status_code = status_code


class OCRPlugin:
    """OCR 插件"""

    def __init__(self, **kwargs):
        self.ocr = None
        self.det_box_thresh = kwargs.get("det_box_thresh", 0.3)
        # 优先使用环境变量；未配置时回退到 system_params.yaml 的 MODEL_DIR
        model_dir = os.getenv("MODEL_DIR") if not os.getenv("RUNNING_IN_DOCKER") else os.getenv("MODEL_DIR_IN_DOCKER")
        if not model_dir:
            model_dir = config.get("MODEL_DIR")
        # 将相对路径解析为项目根目录下的绝对路径
        if model_dir and not os.path.isabs(str(model_dir)):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            model_dir = os.path.abspath(os.path.join(project_root, str(model_dir)))
        self.model_dir_root = model_dir

    def _check_rapid_ocr_availability(self):
        """检查 RapidOCR 模型是否可用"""
        try:
            if not self.model_dir_root:
                raise OCRServiceException(
                    "未配置 MODEL_DIR（或 Docker 下的 MODEL_DIR_IN_DOCKER），无法定位 RapidOCR 模型目录。",
                    "rapid_ocr",
                    "model_dir_not_set",
                )
            model_dir = os.path.join(self.model_dir_root, "SWHL/RapidOCR")
            det_model_dir = os.path.join(model_dir, "PP-OCRv4/ch_PP-OCRv4_det_infer.onnx")
            rec_model_dir = os.path.join(model_dir, "PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx")

            if not os.path.exists(model_dir):
                raise OCRServiceException(
                    f"模型目录不存在: {model_dir}。请下载 SWHL/RapidOCR 模型", "rapid_ocr", "model_not_found"
                )

            if not os.path.exists(det_model_dir) or not os.path.exists(rec_model_dir):
                raise OCRServiceException(
                    f"模型文件缺失。请确认模型文件完整: {det_model_dir}, {rec_model_dir}",
                    "rapid_ocr",
                    "model_incomplete",
                )

            return True

        except Exception as e:
            if isinstance(e, OCRServiceException):
                raise
            else:
                raise OCRServiceException(f"RapidOCR 模型检查失败: {str(e)}", "rapid_ocr", "check_failed")

    def load_model(self):
        """加载 OCR 模型"""
        logger.info("加载 OCR 模型，仅在第一次调用时加载")

        # 先检查模型可用性
        self._check_rapid_ocr_availability()

        model_dir = os.path.join(self.model_dir_root, "SWHL/RapidOCR")
        det_model_dir = os.path.join(model_dir, "PP-OCRv4/ch_PP-OCRv4_det_infer.onnx")
        rec_model_dir = os.path.join(model_dir, "PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx")

        try:
            self.ocr = RapidOCR(det_box_thresh=0.3, det_model_path=det_model_dir, rec_model_path=rec_model_dir)
            logger.info(f"OCR Plugin for det_box_thresh = {self.det_box_thresh} loaded.")
        except Exception as e:
            raise OCRServiceException(f"RapidOCR 模型加载失败: {str(e)}", "rapid_ocr", "load_failed")

    def process_image(self, image, params=None):
        """
        对单张图像执行 OCR 并提取文本

        Args:
            image: 图像数据，支持多种格式：
                  - str: 图像文件路径
                  - PIL.Image: PIL 图像对象
                  - numpy.ndarray: numpy 图像数组
            params: 参数
        Returns:
            str: 提取的文本内容
        """
        # 确保模型已加载
        if self.ocr is None:
            self.load_model()

        # 处理不同类型的输入图像
        try:
            if isinstance(image, str):
                # 图像路径直接传递给 OCR 处理
                image_path = image
                is_temp_file = False
            else:
                # 创建临时文件
                is_temp_file = True
                image_path = self._create_temp_image_file(image)

            # 执行 OCR
            start_time = time.time()
            result, _ = self.ocr(image_path)
            processing_time = time.time() - start_time

            # 清理临时文件
            if is_temp_file and os.path.exists(image_path):
                os.remove(image_path)

            # 提取文本
            if result:
                text = "\n".join([line[1] for line in result])
                log_ocr_request("rapid_ocr", image_path, True, processing_time)
                return text
            else:
                log_ocr_request("rapid_ocr", image_path, False, processing_time, "OCR 未能识别出文本内容")
                return ""

        except Exception as e:
            error_msg = f"OCR 处理失败: {str(e)}"
            log_ocr_request("rapid_ocr", image_path, False, 0, error_msg)
            logger.error(error_msg)
            raise OCRServiceException(error_msg, "rapid_ocr", "processing_failed")

    def _create_temp_image_file(self, image):
        """
        将图像数据保存为临时文件

        Args:
            image: PIL.Image 或 numpy.ndarray 格式的图像数据

        Returns:
            str: 临时文件路径
        """
        # 为临时文件创建目录（如果不存在）
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        # 生成临时文件路径
        temp_filename = f"ocr_temp_{uuid.uuid4().hex[:8]}.png"
        image_path = os.path.join(tmp_dir, temp_filename)

        # 根据图像类型保存文件
        if isinstance(image, Image.Image):
            # 保存 PIL 图像对象到临时文件
            image.save(image_path)
        elif isinstance(image, np.ndarray):
            # 将 numpy 数组转换为 PIL 图像并保存
            Image.fromarray(image).save(image_path)
        else:
            raise ValueError("不支持的图像类型，必须是 PIL.Image 或 numpy 数组")

        return image_path

    def process_pdf(self, pdf_path, params=None):
        """
        处理 PDF 文件并提取文本
        :param pdf_path: PDF 文件路径
        :param params: 参数
        :return: 提取的文本
        """

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            images = []

            pdfDoc = fitz.open(pdf_path)
            totalPage = pdfDoc.page_count
            for pg in tqdm(range(totalPage), desc="to images", ncols=100):
                page = pdfDoc[pg]
                rotate, zoom_x, zoom_y = 0, 2, 2
                mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img_pil)

            # 处理每个图像并合并文本
            all_text = []
            for img_path in tqdm(images, desc="to txt", ncols=100):
                text = self.process_image(img_path)
                all_text.append(text)

            logger.debug(f"PDF OCR result: {all_text[:50]}(...) total {len(all_text)} pages.")
            return "\n\n".join(all_text)

        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            return ""

    def process_file_mineru(self, file_path, params=None):
        """
        使用 Mineru OCR 处理文件
        :param file_path: 文件路径
        :param params: 参数
        :return: 提取的文本
        """
        import requests

        from .mineru import parse_doc

        mineru_ocr_uri = os.getenv("MINERU_OCR_URI", "http://localhost:30000")
        mineru_ocr_uri_health = f"{mineru_ocr_uri}/health"

        try:
            # 健康检查
            health_check_response = requests.get(mineru_ocr_uri_health, timeout=5)
            if health_check_response.status_code != 200:
                error_detail = "Unknown error"
                try:
                    error_detail = health_check_response.json()
                except Exception:
                    error_detail = health_check_response.text

                raise OCRServiceException(
                    f"MinerU OCR 服务健康检查失败: {error_detail}", "mineru_ocr", "health_check_failed"
                )

        except Exception as e:
            if isinstance(e, OCRServiceException):
                raise
            raise OCRServiceException(f"MinerU OCR 服务检查失败: {str(e)}", "mineru_ocr", "service_error")

        try:
            start_time = time.time()
            file_path_list = [file_path]
            output_dir = os.path.join(os.getcwd(), "tmp", "mineru_ocr")

            text = parse_doc(file_path_list, output_dir, backend="vlm-sglang-client", server_url=mineru_ocr_uri)[0]

            processing_time = time.time() - start_time
            log_ocr_request("mineru_ocr", file_path, True, processing_time)

            logger.debug(f"Mineru OCR result: {text[:50]}(...) total {len(text)} characters.")
            return text

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"MinerU OCR 处理失败: {str(e)}"
            log_ocr_request("mineru_ocr", file_path, False, processing_time, error_msg)

            raise OCRServiceException(error_msg, "mineru_ocr", "processing_failed")

    def process_file_paddlex(self, pdf_path, params=None):
        """
        使用 PaddleX OCR 处理 PDF 文件
        :param pdf_path: PDF 文件路径
        :param params: 参数
        :return: 提取的文本
        """
        from .paddlex import analyze_document, check_paddlex_health

        paddlex_uri = os.getenv("PADDLEX_URI", "http://localhost:8080")

        try:
            # 健康检查
            health_check_response = check_paddlex_health(paddlex_uri)
            if not health_check_response.ok:
                error_detail = "Unknown error"
                try:
                    error_detail = health_check_response.json()
                except Exception:
                    error_detail = health_check_response.text

                raise OCRServiceException(
                    f"PaddleX OCR 服务健康检查失败: {error_detail}", "paddlex_ocr", "health_check_failed"
                )
        except Exception as e:
            if isinstance(e, OCRServiceException):
                raise
            raise OCRServiceException(f"PaddleX OCR 服务检查失败: {str(e)}", "paddlex_ocr", "service_error")

        try:
            start_time = time.time()
            result = analyze_document(pdf_path, base_url=paddlex_uri)
            processing_time = time.time() - start_time

            if not result["success"]:
                error_msg = f"PaddleX OCR 处理失败: {result['error']}"
                log_ocr_request("paddlex_ocr", pdf_path, False, processing_time, error_msg)

                raise OCRServiceException(error_msg, "paddlex_ocr", "processing_failed")

            log_ocr_request("paddlex_ocr", pdf_path, True, processing_time)
            return result["full_text"]

        except Exception as e:
            if isinstance(e, OCRServiceException):
                raise
            processing_time = time.time() - start_time if "start_time" in locals() else 0
            error_msg = f"PaddleX OCR 处理失败: {str(e)}"
            log_ocr_request("paddlex_ocr", pdf_path, False, processing_time, error_msg)

            raise OCRServiceException(error_msg, "paddlex_ocr", "processing_failed")

    def process_file_mineru_api(self, file_path, params=None):
        """
        调用 MinerU Agent 轻量解析 API 处理文件（免登录签名上传模式）
        官方文档: https://mineru.net/api/v1/agent/parse/file

        :param file_path: 本地文件绝对路径
        :param params: 参数（如 language="en", page_range="1-10"）
        :return: 提取的文本（Markdown）
        """
        import requests

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"待处理文件不存在: {file_path}")

        base_url = "https://mineru.net/api/v1/agent"
        file_name = os.path.basename(file_path)
        start_time = time.time()

        # 可选参数，默认英文（如 arXiv 论文）
        language = params.get("language", "en") if params else "en"
        page_range = params.get("page_range") if params else None

        try:
            logger.info(f"[MinerU Agent API] 开始处理文件: {file_name}")

            # ==========================================
            # 第一步：请求签名上传 URL 和 Task ID
            # ==========================================
            req_data = {
                "file_name": file_name,
                "language": language,
            }
            if page_range:
                req_data["page_range"] = page_range

            init_resp = requests.post(f"{base_url}/parse/file", json=req_data, timeout=10)
            init_result = init_resp.json()

            if init_result.get("code") != 0:
                raise ValueError(f"获取上传链接失败: {init_result.get('msg')}")

            task_id = init_result["data"]["task_id"]
            file_url = init_result["data"]["file_url"]
            logger.debug(f"[MinerU] 获取任务成功 Task ID: {task_id}")

            # ==========================================
            # 第二步：将本地文件 PUT 到 OSS 签名地址
            # ==========================================
            logger.debug("[MinerU] 正在上传文件到云端...")
            with open(file_path, "rb") as f:
                put_resp = requests.put(file_url, data=f, timeout=60)
                if put_resp.status_code not in (200, 201):
                    raise ValueError(f"文件上传云端失败，HTTP 状态码: {put_resp.status_code}")

            logger.info("[MinerU] 文件上传成功，等待云端解析...")

            # ==========================================
            # 第三步：轮询查询解析结果
            # ==========================================
            timeout = 300  # 最大等待 5 分钟
            interval = 5  # 每 5 秒查一次

            state_labels = {
                "pending": "排队中",
                "running": "解析中",
                "waiting-file": "等待文件同步",
            }

            while time.time() - start_time < timeout:
                poll_resp = requests.get(f"{base_url}/parse/{task_id}", timeout=10)
                poll_result = poll_resp.json()

                if poll_result.get("code") != 0:
                    raise ValueError(f"查询任务状态失败: {poll_result.get('msg')}")

                state = poll_result["data"]["state"]
                elapsed = int(time.time() - start_time)

                if state == "done":
                    markdown_url = poll_result["data"]["markdown_url"]
                    logger.info(f"[MinerU] 解析完成（耗时 {elapsed}s），正在下载 Markdown 内容...")

                    md_resp = requests.get(markdown_url, timeout=30)
                    md_resp.raise_for_status()
                    md_text = md_resp.content.decode("utf-8")

                    log_ocr_request("mineru_agent_api", file_path, True, elapsed)
                    return md_text

                if state == "failed":
                    err_msg = poll_result["data"].get("err_msg", "未知错误")
                    err_code = poll_result["data"].get("err_code", "无")
                    raise ValueError(f"云端解析失败 [Code:{err_code}]: {err_msg}")

                current_status = state_labels.get(state, state)
                logger.debug(f"[MinerU] 耗时 {elapsed}s - 当前状态: {current_status}...")
                time.sleep(interval)

            raise TimeoutError(f"轮询超时（{timeout}s），任务未在预期时间内完成。Task ID: {task_id}")

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"MinerU Agent API 处理异常: {str(e)}"
            log_ocr_request("mineru_agent_api", file_path, False, processing_time, error_msg)

            if "429" in str(e):
                logger.error("触发了 MinerU 的 IP 限频防滥用机制，请稍后再试。")

            raise OCRServiceException(error_msg, "mineru_agent_api", "api_processing_failed")

    def process_file_mineru_token(self, file_path, params=None):
        """
        调用 MinerU V4 精度解析 API 处理文件（基于 MinerUV4Client）

        :param file_path: 本地文件绝对路径
        :param params: 动态参数字典（支持 language、enable_formula、is_ocr、model_version 等）
        :return: 提取的文本（Markdown）
        """
        api_key = config.get("MINERU_API_KEY") or os.getenv("MINERU_API_KEY")
        if not api_key:
            raise ValueError("未找到 MinerU API Key，请在环境变量或配置中设置 MINERU_API_KEY。")

        start_time = time.time()
        params = params or {}

        configs = MinerUV4Config(
            language=params.get("language", "en"),
            enable_formula=params.get("enable_formula", True),
            enable_table=params.get("enable_table", True),
            is_ocr=params.get("is_ocr", False),
            model_version=params.get("model_version", "vlm"),
        )

        try:
            logger.info(f"[MinerU V4 API] 开始处理文件: {os.path.basename(file_path)}")

            client = MinerUV4Client(api_key=api_key, config=configs)
            md_text = client.process_file(file_path)

            elapsed = time.time() - start_time
            log_ocr_request("mineru_v4_api", file_path, True, elapsed)

            return md_text

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"MinerU V4 API 处理异常: {str(e)}"

            log_ocr_request("mineru_v4_api", file_path, False, processing_time, error_msg)

            if "429" in str(e) or "Too Many Requests" in str(e):
                logger.error("触发了 MinerU 的 API 限频机制，并发过高，请稍后再试。")
            elif "401" in str(e) or "Unauthorized" in str(e):
                logger.error("MinerU API Key 鉴权失败，请检查 Key 是否过期或填错。")

            raise OCRServiceException(error_msg, "mineru_v4_api", "api_processing_failed")


def get_state(task_id):
    return GOLBAL_STATE.get(task_id, {})


def plainreader(file_path):
    """读取普通文本文件并返回 text"""
    assert os.path.exists(file_path), "File not found"

    with open(file_path) as f:
        text = f.read()
    return text


if __name__ == "__main__":

    ocr = OCRPlugin()
    text = ocr.process_file_mineru_api(
        r"E:\gitProject\agentpaper\data\knowledge_base_data\chroma_data\kb_kb_6f94f8964d82b604534740d7eff9cd87\uploads\2206.04564v6_TwiBot-22 Towards Graph-Based Twitter Bot Detection.pdf"
    )
