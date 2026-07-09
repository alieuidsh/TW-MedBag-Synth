"""去識別化輸出:對 redact=True 的 PII 區域產生塗黑(或高斯模糊)副本。

資料集因此同時支援兩條研究線:
1. 完整標註影像 → 訓練藥物擷取 + PII 偵測模型。
2. 去識別化影像 → demo「個資不出地端、只送結構化 JSON 上雲」的隱私架構。
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def redact_image(img: Image.Image, regions: list[dict], mode: str = "black") -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for r in regions:
        if not r.get("redact", True):
            continue
        b = r["bbox"]
        box = (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])
        if mode == "blur":
            region = out.crop(box).filter(ImageFilter.GaussianBlur(radius=8))
            out.paste(region, box)
        else:
            draw.rectangle(box, fill=(0, 0, 0))
    return out


def redact_dataset(dataset_dir: Path, out_dir: Path, mode: str = "black") -> int:
    """讀 output/labels/*.json,輸出遮蔽副本到 out_dir,回傳處理張數。"""
    labels_dir = dataset_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for label_path in sorted(labels_dir.glob("*.json")):
        label = json.loads(label_path.read_text(encoding="utf-8"))
        img_path = dataset_dir / label["image"]
        img = Image.open(img_path).convert("RGB")
        redacted = redact_image(img, label["pii_regions"], mode=mode)
        redacted.save(out_dir / Path(label["image"]).name)
        n += 1
    return n
