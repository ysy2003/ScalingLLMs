import os
import io
from dotenv import load_dotenv
import time
import asyncio
import pandas as pd
from typing import Dict, Any, Optional
from PIL import Image, ImageEnhance, ImageFilter

from google import genai
from google.genai.types import HttpOptions
from google.genai import types
from google.cloud import storage

from metrics.robustness import compute_robustness_metrics

load_dotenv()

# ------------------------#
# Global config
# ------------------------#

MAX_NEW_TOKENS = 9000
MODEL_NAME = "gemini-2.5-flash"


PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1"),
)

GCS_CLIENT = storage.Client()

# ------------------------#
# Utility functions
# ------------------------#


def load_prompt_text(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_design2code_uris(
    file_path: str,
    n_images: Optional[int] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    把 dataURI.txt 解析成: number -> {png_uri, html_uri}

    格式和你原来的 gemini.py 一样：
      每行要么以 .png 结尾，要么以 .html 结尾；
      文件名里共同的 number 作为 key。
    """
    png_uris: Dict[str, str] = {}
    html_uris: Dict[str, str] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.endswith(".png"):
                file_name = os.path.basename(line)
                number = file_name.replace(".png", "")
                png_uris[number] = line
            elif line.endswith(".html"):
                file_name = os.path.basename(line)
                number = file_name.replace(".html", "")
                html_uris[number] = line

    # 可选：只取前 n 张调试
    if n_images is not None:
        items = list(png_uris.items())[:n_images]
        png_uris = dict(items)

    pairs: Dict[str, Dict[str, Optional[str]]] = {}
    for number, png_uri in png_uris.items():
        pairs[number] = {
            "png_uri": png_uri,
            "html_uri": html_uris.get(number),
        }
    return pairs


def _read_gcs_bytes(gs_uri: str) -> bytes:
    """
    从 GCS 读取任意对象，返回 bytes。
    例如 gs://bucket/path/to/file.png
    """
    assert gs_uri.startswith("gs://")
    path = gs_uri[5:]  # 去掉 "gs://"
    bucket_name, blob_name = path.split("/", 1)
    blob = GCS_CLIENT.bucket(bucket_name).blob(blob_name)
    return blob.download_as_bytes()


def fetch_image(png_uri: str) -> Image.Image:
    """
    从 GCS / HTTP(S) / 本地路径读取图片。
    """
    if png_uri.startswith("gs://"):
        data = _read_gcs_bytes(png_uri)
        return Image.open(io.BytesIO(data)).convert("RGB")

    if png_uri.startswith(("http://", "https://")):
        resp = requests.get(png_uri, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    # 其他情况当成本地/挂载路径
    return Image.open(png_uri).convert("RGB")


def perturb_image(img: Image.Image, strength: float = 0.05) -> Image.Image:
    """
    对 UI screenshot 做“轻微扰动”。

    strength ∈ [0, 0.3] 控制强度，越大扰动越明显。
    这里模拟现实里的截图噪声：
      - 轻微随机裁剪 + resize 回原尺寸
      - 亮度 / 对比度 jitter
      - 高斯模糊一点点
    """
    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    strength = max(0.0, min(0.5, float(strength)))

    # 1) 边缘随机裁剪再 resize
    max_crop = int(min(w, h) * strength)
    if max_crop > 0:
        import random

        left = random.randint(0, max_crop)
        top = random.randint(0, max_crop)
        right = w - random.randint(0, max_crop)
        bottom = h - random.randint(0, max_crop)
        if right > left and bottom > top:
            img = img.crop((left, top, right, bottom)).resize((w, h), Image.BICUBIC)

    # 2) 亮度 / 对比度 jitter
    b_factor = 1.0 + strength * 0.5
    c_factor = 1.0 + strength * 0.5
    img = ImageEnhance.Brightness(img).enhance(b_factor)
    img = ImageEnhance.Contrast(img).enhance(c_factor)

    # 3) 轻微高斯模糊
    if strength > 0:
        radius = strength * 2.0
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    return img


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


async def generate_html_from_image_bytes(
    img_bytes: bytes,
    prompt_text: str,
    model: str = MODEL_NAME,
) -> Dict[str, Any]:
    """
    用 inline bytes + text prompt 调 Gemini。
    返回:
      {
        "html": str,
        "usage": {token stats...},
      }
    """
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=MAX_NEW_TOKENS,
    )
    try:
        # 用线程池把同步调用包装成“异步”
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=[
                prompt_text,
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            ],
            config=generate_content_config,
        )
    except Exception as e:
        print(f"[ERROR] generate_content failed: {e}")
        return {
            "html": "",
            "usage": {
                "prompt_token_count": None,
                "candidates_token_count": None,
                "thoughts_token_count": None,
                "total_token_count": None,
            },
        }

    html = response.text or ""
    usage_md = getattr(response, "usage_metadata", None)

    usage = {
        "prompt_token_count": getattr(usage_md, "prompt_token_count", None) if usage_md else None,
        "candidates_token_count": getattr(usage_md, "candidates_token_count", None) if usage_md else None,
        "thoughts_token_count": getattr(usage_md, "thoughts_token_count", None) if usage_md else None,
        "total_token_count": getattr(usage_md, "total_token_count", None) if usage_md else None,
    }

    return {"html": html, "usage": usage}


def read_html_from_uri(html_uri: Optional[str]) -> str:
    if not html_uri:
        return ""
    try:
        if html_uri.startswith("gs://"):
            data = _read_gcs_bytes(html_uri)
            # 按 utf-8 解码，多余的错误直接忽略掉
            return data.decode("utf-8", errors="ignore")

        if html_uri.startswith(("http://", "https://")):
            resp = requests.get(html_uri, timeout=30)
            resp.raise_for_status()
            return resp.text

        # 其他情况：本地文件
        with open(html_uri, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] failed to read HTML from {html_uri}: {e}")
        return ""


def compute_design2code_metrics(pred_html: str, gt_html_uri: Optional[str]) -> Dict[str, float]:
    """
    一个很简单的占位版本 metric:
      - visual_fidelity: token 级 Jaccard
      - structural_alignment: HTML tag 名字 Jaccard

    你如果已经有正式的 Design2Code metric，可以直接改掉这个函数，
    保证返回里有 "visual_fidelity" 和 "structural_alignment" 两个 key 即可。
    """
    ref_html = read_html_from_uri(gt_html_uri)

    # Token-level Jaccard over raw text
    def tokenize(s: str) -> set[str]:
        s = (
            s.replace("<", " ")
            .replace(">", " ")
            .replace("/", " ")
            .replace("=", " ")
        )
        return set(t for t in s.split() if t)

    pred_tokens = tokenize(pred_html)
    ref_tokens = tokenize(ref_html)

    inter = len(pred_tokens & ref_tokens)
    union = len(pred_tokens | ref_tokens) or 1
    visual_fidelity = inter / union

    # Tag-level Jaccard over HTML tag names
    import re

    tag_pattern = re.compile(r"<\s*([a-zA-Z0-9]+)")
    pred_tags = set(tag_pattern.findall(pred_html.lower()))
    ref_tags = set(tag_pattern.findall(ref_html.lower()))
    inter_t = len(pred_tags & ref_tags)
    union_t = len(pred_tags | ref_tags) or 1
    structural_alignment = inter_t / union_t

    return {
        "visual_fidelity": visual_fidelity,
        "structural_alignment": structural_alignment,
    }


async def process_single_example(
    number: str,
    png_uri: str,
    html_uri: Optional[str],
    prompt_text: str,
    predictions_dir_clean: str,
    predictions_dir_perturbed: str,
    perturb_strength: float = 0.05,
) -> Dict[str, Any]:
    """
    对一张 UI screenshot:
      1) 用原图跑 Gemini
      2) 用扰动图跑 Gemini
      3) 对两次输出算 Design2Code metric
      4) 用 robustness.py 算 drop
    """
    print(f"[INFO] {number}: loading image from {png_uri}")
    img_clean = fetch_image(png_uri)

    # ----- Clean -----
    clean_bytes = image_to_bytes(img_clean, fmt="PNG")
    t0 = time.time()
    out_clean = await generate_html_from_image_bytes(clean_bytes, prompt_text)
    latency_clean = time.time() - t0

    html_clean = out_clean["html"]
    usage_clean = out_clean["usage"]

    # 保存 clean 预测 HTML
    clean_html_path = os.path.join(predictions_dir_clean, f"{number}.html")
    if html_clean:
        with open(clean_html_path, "w", encoding="utf-8") as f:
            f.write(html_clean)

    # ----- Perturbed -----
    img_pert = perturb_image(img_clean, strength=perturb_strength)
    pert_bytes = image_to_bytes(img_pert, fmt="PNG")
    t1 = time.time()
    out_pert = await generate_html_from_image_bytes(pert_bytes, prompt_text)
    latency_pert = time.time() - t1

    html_pert = out_pert["html"]
    usage_pert = out_pert["usage"]

    pert_html_path = os.path.join(predictions_dir_perturbed, f"{number}.html")
    if html_pert:
        with open(pert_html_path, "w", encoding="utf-8") as f:
            f.write(html_pert)

    # ----- Metrics -----
    clean_metrics = compute_design2code_metrics(html_clean, html_uri)
    pert_metrics = compute_design2code_metrics(html_pert, html_uri)

    robust = compute_robustness_metrics(clean_metrics, pert_metrics)

    result: Dict[str, Any] = {
        "number": number,
        "png_uri": png_uri,
        "html_uri": html_uri,
        "clean_html_path": clean_html_path,
        "perturbed_html_path": pert_html_path,
        "visual_fidelity_clean": clean_metrics.get("visual_fidelity"),
        "visual_fidelity_perturbed": pert_metrics.get("visual_fidelity"),
        "structural_alignment_clean": clean_metrics.get("structural_alignment"),
        "structural_alignment_perturbed": pert_metrics.get("structural_alignment"),
        "visual_fidelity_drop": robust.get("visual_fidelity_drop"),
        "structural_alignment_drop": robust.get("structural_alignment_drop"),
        "latency_clean": latency_clean,
        "latency_perturbed": latency_pert,
        "prompt_token_count_clean": usage_clean.get("prompt_token_count"),
        "candidates_token_count_clean": usage_clean.get("candidates_token_count"),
        "thoughts_token_count_clean": usage_clean.get("thoughts_token_count"),
        "total_token_count_clean": usage_clean.get("total_token_count"),
        "prompt_token_count_perturbed": usage_pert.get("prompt_token_count"),
        "candidates_token_count_perturbed": usage_pert.get("candidates_token_count"),
        "thoughts_token_count_perturbed": usage_pert.get("thoughts_token_count"),
        "total_token_count_perturbed": usage_pert.get("total_token_count"),
    }

    return result


async def run_robustness_eval(
    data_uri_file: str = "gemini/dataURI.txt",
    prompt_path: str = "gemini/prompt.txt",
    out_dir: str = "gemini/results/gemini_robustness",
    logs_file: Optional[str] = "gemini/gemini_robustness_log.jsonl",
    n_images: Optional[int] = None,
    max_concurrent: int = 4,
    perturb_strength: float = 0.05,
) -> pd.DataFrame:
    """
    主入口：
      - 读 Design2Code URIs
      - 对所有图片做 robust eval
      - 返回 per-example metric 的 DataFrame
    """
    os.makedirs(out_dir, exist_ok=True)
    predictions_dir_clean = os.path.join(out_dir, "predictions_clean")
    predictions_dir_perturbed = os.path.join(out_dir, "predictions_perturbed")
    os.makedirs(predictions_dir_clean, exist_ok=True)
    os.makedirs(predictions_dir_perturbed, exist_ok=True)

    prompt_text = load_prompt_text(prompt_path)
    id_to_uris = load_design2code_uris(data_uri_file, n_images=n_images)

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    total = len(id_to_uris)
    lock = asyncio.Lock()

    async def worker(number: str, png_uri: str, html_uri: Optional[str]) -> Dict[str, Any]:
        nonlocal completed
        async with semaphore:
            res = await process_single_example(
                number=number,
                png_uri=png_uri,
                html_uri=html_uri,
                prompt_text=prompt_text,
                predictions_dir_clean=predictions_dir_clean,
                predictions_dir_perturbed=predictions_dir_perturbed,
                perturb_strength=perturb_strength,
            )
            async with lock:
                completed += 1
                print(f"[PROGRESS] {completed}/{total} finished (id={number})")
            return res

    tasks = [
        worker(number, info["png_uri"], info["html_uri"])
        for number, info in id_to_uris.items()
    ]

    print(
        f"[INFO] Starting robustness eval on {len(tasks)} images, "
        f"max_concurrent={max_concurrent}..."
    )
    results: list[Dict[str, Any]] = await asyncio.gather(*tasks)

    df = pd.DataFrame(results)

    # 可选：写 JSONL log
    if logs_file is not None:
        with open(logs_file, "w", encoding="utf-8") as f:
            for row in results:
                f.write(f"{row}\n")

    vf_pdr = df["visual_fidelity_drop"].mean()
    sa_pdr = df["structural_alignment_drop"].mean()
    overall_pdr = 0.5 * (vf_pdr + sa_pdr)

    print("\n=== Robustness: Performance Degradation Rate ===")
    print(f"PDR_VF (visual fidelity):       {vf_pdr:.4f}  ({vf_pdr*100:.2f}%)")
    print(f"PDR_SA (struct alignment):      {sa_pdr:.4f}  ({sa_pdr*100:.2f}%)")
    print(f"PDR_overall (mean of both):     {overall_pdr:.4f}  ({overall_pdr*100:.2f}%)\n")

    # Append PDR results to the log file
    if logs_file is not None:
        with open(logs_file, "a", encoding="utf-8") as f:
            f.write("\n=== Robustness: Performance Degradation Rate ===\n")
            f.write(f"PDR_VF (visual fidelity):       {vf_pdr:.4f}  ({vf_pdr*100:.2f}%)\n")
            f.write(f"PDR_SA (struct alignment):      {sa_pdr:.4f}  ({sa_pdr*100:.2f}%)\n")
            f.write(f"PDR_overall (mean of both):     {overall_pdr:.4f}  ({overall_pdr*100:.2f}%)\n")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gemini-2.5-flash robustness eval on Design2Code"
    )
    parser.add_argument(
        "--data_uri_file",
        type=str,
        default="gemini/dataURI.txt",
        help="Path to dataURI.txt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="gemini/prompt.txt",
        help="Prompt file for Gemini",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="robustness/gemini/results/gemini_robustness",  # 修改默认输出路径
        help="Output directory",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="gemini/gemini_robustness_log.jsonl",
        help="JSONL log file",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=2,  # 修改默认值为 5
        help="Limit number of images (for debugging)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=4,
        help="Max concurrent Gemini calls",
    )
    parser.add_argument(
        "--perturb_strength",
        type=float,
        default=0.05,
        help="Perturbation strength in [0, 0.3]",
    )

    args = parser.parse_args()

    df_results = asyncio.run(
        run_robustness_eval(
            data_uri_file=args.data_uri_file,
            prompt_path=args.prompt,
            out_dir=args.out_dir,
            logs_file=args.logs,
            n_images=args.n_images,
            max_concurrent=args.max_concurrent,
            perturb_strength=args.perturb_strength,
        )
    )

    excel_path = os.path.join(args.out_dir, "gemini_robustness_results.xlsx")
    df_results.to_excel(excel_path, index=False)
    print(
        f"[DONE] {len(df_results)} images evaluated, results saved to {excel_path}"
    )
