# TW-MedBag-Synth — 台灣合成藥袋 OCR 資料集生成器

程式化生成「台灣醫院藥袋」合成影像 + 結構化標註,供繁體中文醫療文件理解
(OCR、資訊擷取、去識別化)模型的訓練與評測使用。
**全部為合成資料,無任何真實病人個資。**

輸入:模擬病人 × 藥物處方的組合。
輸出:**(藥袋影像 PNG, 標註 JSON)** 配對,標註同時包含:

1. **藥物結構化欄位**(藥名/劑量/頻次/時間,含正規化代碼 QD/BID/AC/PC…)
2. **OCR bounding box**(每個欄位、每行文字的像素座標)
3. **PII 去識別化 ground truth**(姓名/病歷號/身分證等區域 + 遮蔽副本)

一份資料兩用:可同時訓練「藥物擷取模型」與「去識別化模型」。

> 現成範例見 [samples/](samples/):診所熱感紙與醫學中心兩種版型的
> 乾淨版 / 增強變體 / 去識別化版對照,以及對應的標註 JSON。

## 快速開始

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Linux/Mac: .venv/bin/pip
.venv\Scripts\python -m playwright install chromium  # 渲染引擎

# 1. 生成 20 位模擬病人(全假值)
python -m src.generate patients --n 20 --out output/patients.json

# 2. 處方 → 渲染 → 標註 → augmentation(每張乾淨圖另加 3 個變體)
python -m src.generate build --aug-per-sample 3 --out output

# 3. 產出 PII 塗黑副本
python -m src.generate deid --in output --out output/redacted

# 4. 驗證:全部 label 過 pydantic schema、bbox 落在畫面內
python -m src.generate validate --in output
```

預設規模:20 位病人 × 每人 3~12 種處方 ≈ 150 筆處方 × (1 乾淨 + 3 變體) ≈ **600 張影像**。
調 `--aug-per-sample 5` 可到 900+ 張;`--n` 加病人數可線性放大。

Linux 使用者請先安裝中文字型:`sudo apt install fonts-noto-cjk`。

## 藥物資料來源(公開、可溯源)

病人為 [Faker](https://faker.readthedocs.io/) 合成、無需出處;
**藥物欄位全部可溯源至政府開放資料**(政府資料開放授權條款第 1 版,
可自由重製再利用,僅需標示來源):

| # | 資料集 | 提供機關 | 用途 | 下載 |
|---|---|---|---|---|
| A | [全部藥品許可證資料集](https://data.gov.tw/dataset/9122) | 食藥署 | 中文品名 / 英文品名 / 劑型 / 適應症 | `https://data.fda.gov.tw/data/opendata/export/36/csv`(ZIP,每 7 日更新) |
| B | [藥品外觀資料集](https://data.gov.tw/dataset/9120) | 食藥署 | 藥袋「外觀描述」欄(形狀/顏色/刻痕/標記) | `https://data.fda.gov.tw/data/opendata/export/42/csv`(ZIP) |
| C | [健保用藥品項查詢項目檔](https://data.gov.tw/dataset/23715) | 健保署 | 健保給付確認、規格、健保代碼 | 資料集頁 CSV(每月更新) |
| D | [健保藥品申報量](https://data.gov.tw/dataset/22131) | 健保署 | 「台灣常見用藥」的客觀選取依據(年度申報量) | 資料集頁(每年) |

**選藥方法論**:以 D 檔年度申報量排序、成分去重取前段,篩除住院專用品項、
補入居家常見特殊劑型(舌下錠/吸入劑),取剛好 100 種(見 `config/drugs.yaml`)。

**溯源工具**:執行 `python scripts/fetch_formulary.py` 會自動下載 A+B 資料集,
逐筆核對 100 種藥名並回填 `license_no`(許可證字號)與官方外觀描述,
輸出 `config/drugs_verified.yaml` —— 任何人都能驗證每個藥名確實存在於公開資料。

> ⚠️ `typical_freq` / `typical_timing`(QD/BID、飯前飯後)在公開資料中僅有
> 自由文字「用法用量」,無標準化代碼;此二欄為依原文**人工正規化**之結果,
> 僅為讓合成藥袋看起來真實的印刷內容範例,**非用藥建議或臨床指引**。

## 標註格式

每張影像對應一個 `output/labels/twmedbag_XXXXX.json`(pydantic 驗證,
schema 見 [src/models.py](src/models.py)):

```jsonc
{
  "sample_id": "twmedbag_00042",
  "image": "images/twmedbag_00042.png",
  "image_size": [640, 902],
  "hospital_template": "generic_v1",
  "is_synthetic": true,
  "patient_ref": "P007",
  "pii_regions": [           // 去識別化 ground truth
    {"field": "patient_name", "bbox": {"x":120,"y":88,"w":64,"h":22},
     "text": "王大明", "redact": true}
  ],
  "prescription": {          // 藥物結構化欄位(一袋一藥)
    "drug_name_zh": "脈優", "drug_name_en": "Amlodipine",
    "strength": "5mg", "frequency_code": "QD", "frequency_text": "每日一次",
    "timing_code": "PC", "timing_text": "飯後", "total_qty": 28.0,
    "field_bboxes": {"drug_name": {...}, "frequency": {...},
                     "timing": {...}, "dose": {...}}
  },
  "ocr_lines": [ {"text": "藥品名稱:脈優 Amlodipine 5mg 錠", "bbox": {...}} ],
  "augmentation": ["perspective", "thermal_fade"]
}
```

正規化代碼(QD/BID/AC/PC…)方便下游應用(排程、統計、查詢)直接使用;
若需展開為實際時間點由應用端處理,資料集刻意**不寫死** 08:00/12:00/18:00。

## 免手標 bbox 的原理(核心賣點)

HTML 版型中每個欄位包 `data-field` 元素、每行文字包 `data-line`:

```html
<span data-field="drug_name" data-pii="false">脈優 Amlodipine 5mg</span>
<span data-field="patient_name" data-pii="true">王大明</span>
```

Playwright 渲染後直接以 `getBoundingClientRect()` 取像素座標
(`device_scale_factor=1`,CSS 像素 = 影像像素)。
圖是我們生成的,標註本來就有 —— **零人工標註成本**。

## Augmentation

模擬手機翻拍藥袋的常見劣化情況(見 [src/augment.py](src/augment.py)):

- **光度類**(bbox 不變):亮度、對比、熱感紙褪色、局部反光、雜訊、JPEG 壓縮、手震模糊
- **幾何類**(記錄 homography,**bbox 四角同步變換後重算外接框**):透視、小角度旋轉

> 增強效果以名稱 registry 組織,可自行替換或擴充效果函式。

## 專案結構

```
config/     drugs.yaml(100 種藥物 formulary)、templates.yaml、enums.py
src/        models.py(schema)、patients.py、prescriptions.py、render.py、
            augment.py、deid.py、labeler.py、generate.py(CLI)
templates/  generic_v1.html(醫學中心 A5)、clinic_v1.html(診所熱感紙)
scripts/    fetch_formulary.py(開放資料下載 + 藥名溯源核對)
tests/      test_schema.py
output/     images/、labels/、redacted/、dataset.json(不進版控)
```

`dataset.json` 的 train/val/test split **依病人切**(同一病人不跨 split),避免資料洩漏。

## 研究倫理與授權

- 本資料集**全部為程式合成**:病人姓名為 Faker 假值;身分證字號為
  「格式正確但未套檢查碼」的亂數(絕大多數為無效號碼,刻意避免與真實證號碰撞);
  醫院/診所名稱皆為虛構。**不對應任何真實個人,無《個人資料保護法》疑慮。**
- 藥物名稱/劑型/適應症取自上表政府開放資料,依政府資料開放授權條款第 1 版
  標示來源;僅供 OCR/資訊擷取模型訓練,**非臨床用藥建議**。
- 含管制藥品(安眠/鎮靜劑)名稱僅作為辨識標的收錄——它們確實常見於
  高齡病人藥袋——與真實處方無關,同樣套用完整 PII 去識別化流程。
- 本專案致敬並在地化 Kaggle *Synthetic Medical Prescription OCR Dataset*
  的合成資料精神(純合成、無隱私風險、ground truth 免人工標註)。
- 授權:程式碼 MIT;生成之資料集建議以 **CC BY 4.0** 發布。

## VLM 基準測試(初步)

`scripts/test_vlm.py`:把藥袋影像丟給 VLM,要求輸出固定 10 欄位 JSON,
與 ground truth 逐欄比對。在 4 張樣本(2 乾淨 + 2 增強)上的初步結果:

| 模型(NVIDIA NIM 免費 API) | JSON 可解析 | 欄位正確率 |
|---|---|---|
| nvidia/nemotron-nano-12b-v2-vl | 4/4 | 75% |
| meta/llama-3.2-90b-vision-instruct | 4/4 | 70% |

兩個共通弱點:**PRN(需要時)的頻次代碼**常被誤判成 QD/TID;
**中文商品名**易回整行文字或幻覺音譯(圖上印「必康平」卻回「泰樂平」)。

**延伸**:腳本支援任何 OpenAI 相容端點——用 Ollama 在本地 GPU 跑
`qwen2.5vl:7b` 之類的中文強項 VLM,比較地端小模型與雲端大模型的差距:

```bash
ollama pull qwen2.5vl:7b
python scripts/test_vlm.py --model qwen2.5vl:7b --api-base http://localhost:11434/v1
```

個資不出機器的本地推論,正好對應本資料集的去識別化研究線。

**微調**:`python scripts/export_sft.py` 會把標註轉成 LLaMA-Factory 相容的
多模態 SFT 資料(sharegpt 格式,train/val/test 依病人切分),
可直接拿去 QLoRA 微調 Qwen2.5-VL 等模型,對照零樣本的欄位正確率。

## 延伸研究方向

1. **OCR fine-tune**:用 `ocr_lines` 訓練/評測 PaddleOCR、TrOCR 於繁中醫療場景。
2. **版面理解**:用 `field_bboxes` 訓練 LayoutLM / Donut 類模型直接輸出結構化 JSON。
3. **PII 偵測**:用 `pii_regions` 訓練去識別化模型,對照 `output/redacted/` 驗收。
4. **Sim-to-real gap**:蒐集(自己的)真實藥袋照片少量標註,量測合成→真實的落差,
   反推 augmentation 該加強哪些效果。
