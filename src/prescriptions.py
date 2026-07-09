"""病人 × 藥物 → 處方組合,並產生藥袋要印的所有文字。

這裡決定的內容 = ground truth:
渲染只是把這些字「畫」出來,所以標註天生正確,零人工標註成本。
"""
import random
import zlib
from datetime import date, timedelta

import yaml

from config.enums import FREQUENCY, TIMING, FORM_UNIT

# ── 外觀描述(合成用;正式資料請執行 scripts/fetch_formulary.py
#    以食藥署「藥品外觀資料集」回填真實外觀)────────────────
_SHAPES = ["圓形", "橢圓形", "長條形"]
_COLORS = ["白色", "淡黃色", "粉紅色", "淡藍色", "橘色", "淡綠色"]

# ── 警語:依藥名 / 適應症關鍵字挑選 ───────────────────────
_WARNING_RULES: list[tuple[str, str]] = [
    ("Warfarin",  "維持固定飲食習慣,避免大量食用深綠色蔬菜;定期回診監測凝血功能"),
    ("Nitroglycerin", "發作時舌下含服;5分鐘未緩解可再含1顆,最多3顆,仍未緩解請立即就醫"),
    ("Alendronate", "早晨空腹以大量開水吞服,服藥後30分鐘內勿平躺、勿進食"),
    ("Acetaminophen", "24小時內總量不可超過4000毫克;避免併用其他含乙醯胺酚藥品"),
    ("Colchicine", "依醫囑服用;出現腹瀉請停藥並回診"),
    ("Amlodipine", "避免與葡萄柚汁併服"),
    ("Nifedipine", "避免與葡萄柚汁併服"),
    ("抗凝血",     "服藥期間避免劇烈碰撞;如有異常出血請立即回診"),
    ("降血脂",     "避免與葡萄柚汁併服"),
    ("助眠",       "服藥後請勿開車或操作機械;可能產生嗜睡"),
    ("安眠",       "服藥後請勿開車或操作機械;可能產生嗜睡"),
    ("抗焦慮",     "服藥後請勿開車或操作機械;可能產生嗜睡"),
    ("精神",       "服藥後請勿開車或操作機械;可能產生嗜睡"),
    ("消炎止痛",   "腸胃不適者請與食物併服;如出現黑便請回診"),
    ("抗生素",     "請務必服用完整療程,勿自行停藥"),
    ("降血糖",     "注意低血糖症狀(冒冷汗、心悸、頭暈);外出請隨身攜帶糖果"),
    ("胰島素",     "注射部位請輪替;注意低血糖症狀"),
    ("利尿",       "建議早上服用,避免夜間頻尿影響睡眠"),
    ("甲狀腺",     "須空腹服用,與其他藥物間隔至少4小時"),
    ("支氣管擴張", "使用前搖勻;急性發作使用後若未緩解請儘速就醫"),
    ("補鐵",       "可能使糞便變黑屬正常現象;避免與茶、咖啡併服"),
    ("攝護腺",     "可能引起姿勢性低血壓,起身時請放慢動作"),
    ("抗組織胺",   "可能產生嗜睡,開車前請留意自身反應"),
]
_WARNING_DEFAULT = [
    "請依醫囑按時服藥,勿自行增減劑量",
    "藥品請存放於陰涼乾燥處,避免兒童取得",
    "如出現皮疹、搔癢等過敏症狀請停藥並回診",
]


def age_at(birth_iso: str, on_iso: str) -> int:
    """實歲:藥袋上的年齡以「領藥日期」計算,與出生日期完全自洽。"""
    b, d = date.fromisoformat(birth_iso), date.fromisoformat(on_iso)
    return d.year - b.year - ((d.month, d.day) < (b.month, b.day))


def load_drugs(path) -> list[dict]:
    data = yaml.safe_load(open(path, encoding="utf-8"))
    return data["drugs"]


def _stable_rng(key: str) -> random.Random:
    # 以 crc32 取代內建 hash():內建 hash 每次執行加鹽,結果不可重現
    return random.Random(zlib.crc32(key.encode("utf-8")))


def appearance_text(drug: dict) -> str:
    """同一種藥固定同一外觀(由學名決定),模擬真實藥袋的外觀描述欄。"""
    rng = _stable_rng(drug["name_en"])
    form = drug["form"]
    if form == "糖漿":
        return "琥珀色透明糖漿,瓶裝"
    if form == "吸入劑":
        return "定量噴霧吸入器,藍白色塑膠瓶身"
    if form == "注射筆":
        return "預填充式注射筆,附刻度視窗"
    color = rng.choice(_COLORS)
    if form == "膠囊":
        color2 = rng.choice([c for c in _COLORS if c != color])
        return f"{color}/{color2}膠囊"
    shape = rng.choice(_SHAPES)
    mark = drug["strengths"][0].replace("mg", "").replace("mcg", "")
    kind = {"腸溶錠": "腸溶錠", "緩釋錠": "緩釋錠",
            "舌下錠": "舌下錠", "發泡錠": "發泡錠"}.get(form, "錠")
    return f"{color}{shape}{kind},一面刻 {mark}"


def warning_text(drug: dict, rng: random.Random) -> str:
    for key, text in _WARNING_RULES:
        if key in drug["name_en"] or key in drug["indication"]:
            return text
    return rng.choice(_WARNING_DEFAULT)


def _dose_and_duration(drug: dict, rng: random.Random) -> tuple[float, int, float, int | None]:
    """回傳 (每次量, 天數, 總量, PRN 每日上限)。"""
    freq = drug["typical_freq"]
    form = drug["form"]
    per_day = FREQUENCY[freq][1]

    if form == "糖漿":
        dose = float(rng.choice([10, 15]))
    elif form == "吸入劑":
        dose = float(rng.choice([1, 2]))
    elif form == "注射筆":
        dose = float(rng.choice([10, 14, 18, 22]))
    elif drug["name_en"] in ("Warfarin", "Digoxin"):
        dose = rng.choice([0.5, 1.0])
    else:
        dose = float(rng.choices([1, 2], weights=[8, 2])[0])

    if drug["indication"] == "抗生素":
        duration = rng.choice([3, 5, 7])
    elif freq == "PRN":
        duration = 14
    else:
        duration = rng.choice([7, 14, 28, 28, 30])

    if freq == "QW":
        total = dose * max(round(duration / 7), 1)
        return dose, duration, total, None
    if freq == "PRN":
        max_daily = 4 if drug["name_en"] == "Acetaminophen" else rng.choice([2, 3])
        total = float(rng.choice([10, 15, 20]))
        return dose, duration, total, max_daily
    total = dose * (per_day or 1) * duration
    return dose, duration, total, None


def _dosage_parts(drug: dict, dose: float, unit: str,
                  max_daily: int | None) -> list[dict]:
    """用法用量行拆成帶欄位標記的片段,f ∈ {frequency, dose, timing, None}。

    版型會把有 f 的片段包成 data-field 子元素,
    使 frequency/timing/dose 各自擁有精確 bbox(spec §7.2 field_bboxes)。
    """
    freq = drug["typical_freq"]
    freq_zh = FREQUENCY[freq][0]
    timing_zh = TIMING[drug["typical_timing"]]
    form = drug["form"]

    if form == "注射筆":
        return [{"t": timing_zh, "f": "timing"}, {"t": "皮下注射 ", "f": None},
                {"t": f"{dose:g} 單位", "f": "dose"}]
    if form == "吸入劑":
        return [{"t": timing_zh, "f": "timing"}, {"t": "吸入 ", "f": None},
                {"t": f"{dose:g} 下", "f": "dose"},
                {"t": ",每日不超過 8 下", "f": None}]
    if freq == "PRN":
        return [{"t": timing_zh, "f": "timing"}, {"t": "服用 ", "f": None},
                {"t": f"{dose:g} {unit}", "f": "dose"},
                {"t": f",每日不超過 {max_daily} 次", "f": None}]
    if freq == "HS":
        return [{"t": freq_zh, "f": "frequency"}, {"t": "服用 ", "f": None},
                {"t": f"{dose:g} {unit}", "f": "dose"}]
    if freq == "QW":
        return [{"t": freq_zh, "f": "frequency"}, {"t": ",", "f": None},
                {"t": timing_zh, "f": "timing"}, {"t": ",每次 ", "f": None},
                {"t": f"{dose:g} {unit}", "f": "dose"}]
    return [{"t": freq_zh, "f": "frequency"}, {"t": ",每次 ", "f": None},
            {"t": f"{dose:g} {unit}", "f": "dose"}, {"t": ",", "f": None},
            {"t": timing_zh, "f": "timing"}, {"t": "服用", "f": None}]


def make_prescriptions(patients: list[dict], drugs: list[dict],
                       seed: int = 42) -> list[dict]:
    """每位病人隨機 3~12 種藥,一藥一袋 → 回傳攤平的處方清單。"""
    rng = random.Random(seed)
    fake_start = date(2025, 7, 1)
    out = []
    for p in patients:
        k = rng.randint(3, 12)
        chosen = rng.sample(drugs, k)
        dispense = fake_start + timedelta(days=rng.randint(0, 330))
        for idx, drug in enumerate(chosen):
            dose, duration, total, max_daily = _dose_and_duration(drug, rng)
            unit = FORM_UNIT[drug["form"]]
            strength = rng.choice(drug["strengths"])
            parts = _dosage_parts(drug, dose, unit, max_daily)
            out.append({
                "dosage_parts": parts,
                "patient": p,
                "index": idx,
                "drug": drug,
                "strength": str(strength),
                "dose_per_admin": dose,
                "unit": unit,
                "duration_days": duration,
                "total_qty": total,
                "prn": drug["typical_freq"] == "PRN",
                "dispense_date": dispense.isoformat(),
                "dosage_text": "".join(p["t"] for p in parts),
                "total_text": f"{total:g} {unit}(服用 {duration} 天)",
                "appearance_text": appearance_text(drug),
                "warning_text": warning_text(drug, rng),
            })
    return out
