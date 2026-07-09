"""把標註轉成 VLM 微調(SFT)用的多模態對話資料。

輸出 LLaMA-Factory 相容的 sharegpt 格式(messages + images),
train/val/test 依 dataset.json 的病人切分,無資料洩漏:

    python scripts/export_sft.py                 # 讀 output/ → 寫 output/sft_*.jsonl
    python scripts/export_sft.py --data output

每筆格式:
    {"messages": [{"role": "user", "content": "<image>...提示詞..."},
                  {"role": "assistant", "content": "{...ground truth JSON...}"}],
     "images": ["images/twmedbag_00000.png"]}

之後在 LLaMA-Factory 的 dataset_info.json 登錄即可微調 Qwen2.5-VL 等模型
(formatting: sharegpt, columns: messages/images)。
"""
import argparse
import json
from pathlib import Path

# 與評測共用同一份提示詞與欄位定義,訓練/測試才一致
from test_vlm import FIELDS, PROMPT, ground_truth

ROOT = Path(__file__).resolve().parent.parent


def build_record(label_path: Path) -> dict:
    label = json.loads(label_path.read_text(encoding="utf-8"))
    gt = ground_truth(label_path)
    answer = json.dumps({f: gt.get(f) for f in FIELDS}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content": "<image>" + PROMPT},
            {"role": "assistant", "content": answer},
        ],
        "images": [label["image"]],   # 相對於 output/,JSONL 也放 output/
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "output",
                    help="含 labels/ 與 dataset.json 的資料夾")
    args = ap.parse_args()

    manifest = json.loads((args.data / "dataset.json").read_text(encoding="utf-8"))
    for split, sample_ids in manifest["splits"].items():
        records = [build_record(args.data / "labels" / f"{sid}.json")
                   for sid in sample_ids]
        out = args.data / f"sft_{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(records)} 筆 → {out}")


if __name__ == "__main__":
    main()
