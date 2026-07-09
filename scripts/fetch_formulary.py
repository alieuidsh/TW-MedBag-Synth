"""從政府開放資料下載藥品資料,核對/回填 drugs.yaml 的溯源欄位。

資料來源(政府資料開放授權條款第 1 版,可自由重製再利用,需標示來源):
  A. 食藥署「全部藥品許可證資料集」 https://data.gov.tw/dataset/9122
     → 中文品名 / 英文品名 / 劑型 / 適應症 / 許可證字號
  B. 食藥署「藥品外觀資料集」       https://data.gov.tw/dataset/9120
     → 形狀 / 顏色 / 刻痕 / 標記(對應藥袋「外觀描述」欄)
  C. 健保署「健保用藥品項查詢項目檔」 https://data.gov.tw/dataset/23715
     → 健保代碼 / 規格(請自該頁下載 CSV 後以 --nhi-csv 傳入)

用法:
    python scripts/fetch_formulary.py                 # 下載 A+B 並核對
    python scripts/fetch_formulary.py --nhi-csv 健保品項.csv
輸出:
    config/drugs_verified.yaml  (每筆多 license_no / appearance_official 欄)
    data_raw/                   (下載的原始 ZIP/CSV,已列入 .gitignore)
"""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

FDA_LICENSE_URL = "https://data.fda.gov.tw/data/opendata/export/36/csv"  # A
FDA_APPEAR_URL = "https://data.fda.gov.tw/data/opendata/export/42/csv"   # B

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw"


def fetch_fda_csv(url: str, cache_name: str) -> pd.DataFrame:
    """下載食藥署 ZIP(內含單一 CSV)→ DataFrame,存原始檔供重現。"""
    RAW_DIR.mkdir(exist_ok=True)
    cache = RAW_DIR / cache_name
    if cache.exists():
        print(f"  使用快取 {cache}")
        data = cache.read_bytes()
    else:
        print(f"  下載 {url} ...")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        data = r.content
        cache.write_bytes(data)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, encoding="utf-8", dtype=str, low_memory=False)


def find_col(df: pd.DataFrame, keyword: str) -> str:
    """欄位名稱可能隨版本微調,用關鍵字尋找。"""
    for col in df.columns:
        if keyword in col:
            return col
    raise KeyError(f"找不到含「{keyword}」的欄位;實際欄位:{list(df.columns)}")


def match_drug(drug: dict, lic: pd.DataFrame,
               col_zh: str, col_en: str, col_no: str) -> str | None:
    """以 中文品名 包含 name_zh 且 英文品名 開頭符合 name_en 首字 來匹配。"""
    token = drug["name_en"].split("/")[0].split()[0].upper()
    hits = lic[
        lic[col_zh].fillna("").str.contains(drug["name_zh"], regex=False)
        & lic[col_en].fillna("").str.upper().str.contains(token, regex=False)
    ]
    if hits.empty:  # 放寬:只比對中文品名
        hits = lic[lic[col_zh].fillna("").str.contains(drug["name_zh"], regex=False)]
    return None if hits.empty else str(hits.iloc[0][col_no])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nhi-csv", type=Path, default=None,
                    help="健保用藥品項 CSV(自 data.gov.tw/dataset/23715 下載)")
    args = ap.parse_args()

    print("A. 全部藥品許可證資料集")
    lic = fetch_fda_csv(FDA_LICENSE_URL, "fda_license.zip")
    print("B. 藥品外觀資料集")
    app = fetch_fda_csv(FDA_APPEAR_URL, "fda_appearance.zip")

    col_zh = find_col(lic, "中文品名")
    col_en = find_col(lic, "英文品名")
    col_no = find_col(lic, "許可證字號")
    app_no = find_col(app, "許可證字號")

    drugs = yaml.safe_load((ROOT / "config" / "drugs.yaml").read_text(encoding="utf-8"))["drugs"]
    matched = 0
    for drug in drugs:
        license_no = match_drug(drug, lic, col_zh, col_en, col_no)
        drug["license_no"] = license_no
        if license_no:
            matched += 1
            rows = app[app[app_no] == license_no]
            if not rows.empty:
                row = rows.iloc[0]
                parts = [str(row[c]) for c in app.columns
                         for k in ("顏色", "形狀", "刻痕", "標記")
                         if k in c and pd.notna(row[c]) and str(row[c]).strip()]
                if parts:
                    drug["appearance_official"] = " / ".join(parts)
        else:
            print(f"  [未匹配] {drug['name_zh']} {drug['name_en']}(請人工核對)")

    if args.nhi_csv:
        nhi = pd.read_csv(args.nhi_csv, dtype=str, low_memory=False)
        nhi_no = find_col(nhi, "許可證")
        nhi_code = find_col(nhi, "藥品代")
        lookup = nhi.set_index(nhi_no)[nhi_code].to_dict()
        for drug in drugs:
            if drug.get("license_no") in lookup:
                drug["nhi_code"] = lookup[drug["license_no"]]

    out = ROOT / "config" / "drugs_verified.yaml"
    out.write_text(yaml.safe_dump({"drugs": drugs}, allow_unicode=True,
                                  sort_keys=False, width=200), encoding="utf-8")
    print(f"匹配 {matched}/{len(drugs)} 種 → {out}")


if __name__ == "__main__":
    main()
