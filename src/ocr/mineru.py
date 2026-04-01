import requests
import time
import os
import uuid
import logging
import zipfile
from io import BytesIO
from dataclasses import dataclass, asdict
from typing import Optional

import urllib3

# 与下方 requests(..., verify=False) 配套，避免控制台刷屏 InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


@dataclass
class MinerUV4Config:
    """
    MinerU V4 精度解析配置类
    官方文档: https://mineru.net/apiManage/docs
    """
    language: str = "en"
    enable_formula: bool = True
    enable_table: bool = True
    is_ocr: bool = False
    model_version: str = "vlm"  # 推荐高精度大模型模式


class MinerUV4Client:
    """
    MinerU V4 API 客户端：支持全量解压（含图片）
    """

    def __init__(self, api_key: str, config: Optional[MinerUV4Config] = None, timeout_sec: int = 600):
        if not api_key:
            raise ValueError("必须提供 API Key 才能调用 V4 精度解析接口")

        self.api_key = api_key
        self.config = config or MinerUV4Config()
        self.timeout = timeout_sec
        self.base_url = "https://mineru.net/api/v4"
        self.current_pdf_path = None  # 用于内部传递路径

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.direct_proxies = {"http": None, "https": None}

    def process_file(self, file_path: str) -> str:
        """主入口：处理文件并在本地生成结果文件夹"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"待处理文件不存在: {file_path}")

        self.current_pdf_path = file_path
        file_name = os.path.basename(file_path)
        start_time = time.time()
        data_id = str(uuid.uuid4())

        try:
            # 1. 申请上传链接
            batch_id, upload_url = self._init_batch_upload(file_name, data_id)

            # 2. 上传文件
            self._upload_file(file_path, upload_url)

            # 3. 轮询并下载解压
            md_text = self._poll_and_download(batch_id, start_time)

            return md_text

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[MinerU V4] 处理异常 (耗时 {int(elapsed)}s): {str(e)}")
            raise

    def _init_batch_upload(self, file_name: str, data_id: str) -> tuple[str, str]:
        """第一步：通过 /file-urls/batch 申请链接"""
        logger.info(f"[MinerU V4] 正在初始化上传与解析任务: {file_name}")
        payload = asdict(self.config)
        payload["files"] = [{"name": file_name, "data_id": data_id}]

        resp = requests.post(
            f"{self.base_url}/file-urls/batch",
            headers=self.headers,
            json=payload,
            timeout=15,
            verify=False,
            proxies=self.direct_proxies
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 0:
            raise ValueError(f"获取上传链接失败: {result}")

        batch_id = result["data"]["batch_id"]
        upload_url = result["data"]["file_urls"][0]
        return batch_id, upload_url

    def _upload_file(self, file_path: str, upload_url: str):
        """第二步：上传文件"""
        logger.info("[MinerU V4] 正在上传文件至云端存储...")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        for attempt in range(3):
            try:
                resp = requests.put(
                    upload_url,
                    data=file_bytes,
                    verify=False,
                    proxies=self.direct_proxies,
                    timeout=120
                )
                if resp.status_code in (200, 201):
                    logger.info("[MinerU V4] 文件上传成功！系统将自动触发解析。")
                    return
                else:
                    raise ValueError(f"上传失败，HTTP 状态码: {resp.status_code}")
            except requests.exceptions.RequestException:
                logger.warning(f"[MinerU V4] 上传遇到网络波动 (第 {attempt + 1}/3 次尝试)...")
                if attempt == 2:
                    raise
                time.sleep(3)

    def _poll_and_download(self, batch_id: str, start_time: float) -> str:
        """第三步：轮询状态并执行全量解压"""
        logger.info(f"[MinerU V4] 等待云端解析... BatchID: {batch_id}")

        # 确定解压目录：PDF 同级目录下的 {文件名}_extract 文件夹
        pdf_dir = os.path.dirname(self.current_pdf_path)
        pdf_name_stem = os.path.splitext(os.path.basename(self.current_pdf_path))[0]
        extract_path = os.path.join(pdf_dir, f"{pdf_name_stem}_extract")

        interval = 5
        while time.time() - start_time < self.timeout:
            resp = requests.get(
                f"{self.base_url}/extract-results/batch/{batch_id}",
                headers=self.headers,
                timeout=15,
                verify=False,
                proxies=self.direct_proxies
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get("code") != 0:
                raise ValueError(f"查询任务状态失败: {result.get('msg')}")

            extract_results = result["data"].get("extract_result", [])
            if not extract_results:
                time.sleep(interval)
                continue

            file_status = extract_results[0]
            state = file_status.get("state", "").lower()
            elapsed = int(time.time() - start_time)

            if state in ("done", "success", "finished"):
                logger.info(f"[MinerU V4] 解析成功！耗时 {elapsed}s，准备处理结果...")

                zip_url = file_status.get("full_zip_url")
                if zip_url:
                    zip_resp = requests.get(zip_url, timeout=60, verify=False, proxies=self.direct_proxies)
                    zip_resp.raise_for_status()

                    # 创建解压目录并解压
                    os.makedirs(extract_path, exist_ok=True)
                    with zipfile.ZipFile(BytesIO(zip_resp.content)) as z:
                        z.extractall(extract_path)

                    logger.info(f"[MinerU V4] 已解压全部附件（含图片）至: {extract_path}")

                    # 寻找并读取解压后的 MD 文件内容
                    for filename in os.listdir(extract_path):
                        if filename.endswith(".md"):
                            with open(os.path.join(extract_path, filename), "r", encoding="utf-8") as f:
                                return f.read()

                    raise ValueError("压缩包内未找到 .md 文件")

                raise ValueError("云端未返回 full_zip_url")

            elif state in ("failed", "error"):
                raise ValueError(f"云端解析失败: {file_status.get('err_msg', '未知错误')}")

            else:
                logger.info(f"[MinerU V4] 处理中... 当前状态: {state} (耗时 {elapsed}s)")
                time.sleep(interval)

        raise TimeoutError(f"轮询超时 ({self.timeout}s)")


if __name__ == "__main__":
    # 配置你的 PDF 路径
    pdf_path = r"E:\gitProject\agentpaper\data\knowledge_base_data\chroma_data\kb_kb_6f94f8964d82b604534740d7eff9cd87\uploads\2206.08029v1_DIALOG-22 RuATD Generated Text Detection.pdf"

    # API 配置
    API_KEY = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI3MjAwMDUyMyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3NDQ5MDMxMywiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTg5NDA4NDQ1ODciLCJvcGVuSWQiOm51bGwsInV1aWQiOiI4YzBhNGI3Yi1jNjgyLTRkMDAtYjhiNC1kNjcwYWE3ZGY1YjIiLCJlbWFpbCI6ImV5bm9ubGxoNjlAZ21haWwuY29tIiwiZXhwIjoxNzgyMjY2MzEzfQ.urkzIQdiKZIqB9JbC6mRy8RFLk_i_GWleML061Ac_89uEMuEpg8GdUvXxlnr3mPFoJ1SddmChHqaTa2wRncLgg"
    hq_config = MinerUV4Config(
        language="en",
        enable_formula=True,
        enable_table=True,
        is_ocr=False,
        model_version="vlm"  # 使用高精度 VLM 模型
    )

    client = MinerUV4Client(api_key=API_KEY, config=hq_config)

    try:
        # 执行处理
        result_md_content = client.process_file(pdf_path)

        # 保存 Markdown 文件
        save_dir = os.path.dirname(pdf_path)
        file_stem = os.path.splitext(os.path.basename(pdf_path))[0]
        md_save_path = os.path.join(save_dir, f"{file_stem}.md")

        with open(md_save_path, "w", encoding="utf-8") as f:
            f.write(result_md_content)

        print("-" * 30)
        print("✅ 解析大功告成！")
        print(f"📄 Markdown 文件: {md_save_path}")
        print(f"🖼️ 图片及附件目录: {os.path.join(save_dir, f'{file_stem}_extract')}")
        print("-" * 30)

    except Exception as e:
        print(f"❌ 运行失败: {e}")