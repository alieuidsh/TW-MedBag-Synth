"""VLM 藥袋辨識基準測試:影像 → 固定格式 JSON → 與 ground truth 逐欄比對。

用途:驗證「拍藥袋 → VLM → 結構化 JSON」流程的可行性,
並比較不同模型的欄位擷取正確率。

用法(API 金鑰放環境變數,切勿寫進程式碼/版控):
    set NVIDIA_API_KEY=nvapi-xxxx        # PowerShell: $env:NVIDIA_API_KEY='...'
    python scripts/test_vlm.py --model meta/llama-4-maverick-17b-128e-instruct
    python scripts/test_vlm.py --model nvidia/nemotron-nano-12b-v2-vl --ids 00000 00604

本地模型(Ollama 等 OpenAI 相容端點,免金鑰):
    ollama pull qwen2.5vl:7b
    python scripts/test_vlm.py --model qwen2.5vl:7b \
        --api-base http://localhost:11434/v1

金鑰申請:https://build.nvidia.com(每個帳號有免費額度)。
"""
import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import requests

DEFAULT_API_BASE = "https://integrate.api.nvidia.com/v1"
ROOT = Path(__file__).resolve().parent.parent

PROMPT = """你是藥袋資訊擷取系統。請從這張台灣藥袋影像擷取資訊,只輸出一個 JSON 物件,不要任何其他文字、不要 markdown 圍欄。格式:
{"drug_name_zh": "中文藥名(商品名)",
 "drug_name_en": "英文學名(不含劑量)",
 "strength": "劑量規格,如 5mg",
 "dose_per_admin": 每次服用數量(數字),
 "unit": "計量單位,如 顆、mL、下、單位",
 "frequency_code": "頻次代碼,只能是 QD(每日一次)/BID(每日兩次)/TID(每日三次)/QID(每日四次)/QW(每週一次)/HS(每晚睡前)/PRN(需要時) 之一",
 "timing_text": "服藥時間,如 飯前、飯後、睡前、隨餐、早上",
 "duration_days": 服用天數(整數),
 "total_qty": 總量(數字),
 "indication": "適應症"}
影像中找不到的欄位填 null。"""

# 評分欄位與比對方式
_NUM_FIELDS = {"dose_per_admin", "total_qty", "duration_days"}
_LOOSE_FIELDS = {"timing_text", "indication"}          # 互相包含即算對
FIELDS = ["drug_name_zh", "drug_name_en", "strength", "dose_per_admin",
          "unit", "frequency_code", "timing_text", "duration_days",
          "total_qty", "indication"]


def ask_vlm(model: str, image_path: Path, api_key: str, api_base: str,
            timeout: int = 120, retries: int = 2) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
            if r.status_code == 400:
                # 部分舊 NIM 視覺模型只吃「<img> 嵌在文字」格式,fallback 重試
                body["messages"] = [{"role": "user",
                                     "content": f'{PROMPT} <img src="data:image/png;base64,{b64}" />'}]
                r = requests.post(url, headers=headers, json=body, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (requests.Timeout, requests.ConnectionError) as e:
            # NIM 免費層冷啟動可能排隊逾時,重試通常已暖機
            last_err = e
            print(f"    (第 {attempt + 1} 次逾時,重試中…)")
    raise last_err


def extract_json(text: str) -> dict | None:
    """容錯解析:剝 markdown 圍欄後找第一個平衡的 {...}。"""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return None
    return None


def _norm(v) -> str:
    return str(v).strip().lower().replace(" ", "").replace(",", "")


def compare(pred: dict, gt: dict) -> dict[str, bool]:
    result = {}
    for f in FIELDS:
        p, g = pred.get(f), gt.get(f)
        if p is None:
            result[f] = False
        elif f in _NUM_FIELDS:
            try:
                result[f] = abs(float(p) - float(g)) < 1e-6
            except (TypeError, ValueError):
                result[f] = False
        elif f in _LOOSE_FIELDS:
            result[f] = _norm(p) in _norm(g) or _norm(g) in _norm(p)
        elif f == "drug_name_en":
            result[f] = _norm(g) in _norm(p) or _norm(p) in _norm(g)
        else:
            result[f] = _norm(p) == _norm(g)
    return result


def ground_truth(label_path: Path) -> dict:
    rx = json.loads(label_path.read_text(encoding="utf-8"))["prescription"]
    return {f: rx[f] for f in FIELDS if f in rx} | {
        "timing_text": rx["timing_text"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", nargs="*",
                    default=["00000", "00003", "00604", "00607"])
    ap.add_argument("--data", type=Path, default=ROOT / "output",
                    help="含 images/ 與 labels/ 的資料夾")
    ap.add_argument("--api-base", default=DEFAULT_API_BASE,
                    help="OpenAI 相容端點,如 http://localhost:11434/v1(Ollama)")
    ap.add_argument("--verbose", action="store_true", help="印出模型原始回覆")
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "not-needed")
    if args.api_base == DEFAULT_API_BASE and api_key == "not-needed":
        sys.exit("請先設定環境變數 NVIDIA_API_KEY(勿寫進程式碼)")

    totals = {f: 0 for f in FIELDS}
    n_ok_json = 0
    for sid in args.ids:
        img = args.data / "images" / f"twmedbag_{sid}.png"
        lbl = args.data / "labels" / f"twmedbag_{sid}.json"
        gt = ground_truth(lbl)
        try:
            raw = ask_vlm(args.model, img, api_key, args.api_base)
        except requests.HTTPError as e:
            print(f"[{sid}] API 錯誤:{e.response.status_code} {e.response.text[:200]}")
            continue
        except requests.RequestException as e:
            print(f"[{sid}] 連線失敗:{type(e).__name__}")
            continue
        pred = extract_json(raw)
        if args.verbose:
            print(f"--- {sid} raw ---\n{raw}\n")
        if pred is None:
            print(f"[{sid}] 無法解析 JSON")
            continue
        n_ok_json += 1
        marks = compare(pred, gt)
        for f, ok in marks.items():
            totals[f] += ok
        wrong = [f"{f}:{pred.get(f)!r}≠{gt.get(f)!r}"
                 for f, ok in marks.items() if not ok]
        score = sum(marks.values())
        print(f"[{sid}] {score}/{len(FIELDS)}"
              + (f"  錯誤 → {'; '.join(wrong)}" if wrong else "  全對"))

    n = len(args.ids)
    print(f"\n=== {args.model} ===")
    print(f"JSON 可解析率:{n_ok_json}/{n}")
    for f in FIELDS:
        print(f"  {f:<16} {totals[f]}/{n}")
    grand = sum(totals.values())
    print(f"  {'overall':<16} {grand}/{n * len(FIELDS)}"
          f" ({grand / (n * len(FIELDS)):.0%})")


if __name__ == "__main__":
    main()
