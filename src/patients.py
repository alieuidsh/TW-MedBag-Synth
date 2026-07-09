"""生成模擬病人(全假值,不對應任何真實個人)。

隱私設計:
- 姓名由 Faker(zh_TW)產生。
- 身分證字號為「格式正確但未套用檢查碼」的隨機值——絕大多數是無效號碼,
  刻意避免與真實國民的有效證號碰撞;OCR / 去識別化訓練只需要「長得像」。
- 病歷號為 8 位亂數。
"""
import json
import random
import string
from datetime import date, timedelta
from pathlib import Path

from faker import Faker


def _fake_national_id(rng: random.Random, sex: str) -> str:
    letter = rng.choice(string.ascii_uppercase)
    gender_digit = "1" if sex == "男" else "2"
    return letter + gender_digit + "".join(rng.choices(string.digits, k=8))


def _shift_years(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year - n)
    except ValueError:          # 2/29 → 2/28
        return d.replace(year=d.year - n, day=28)


def _birth_date_from_age(rng: random.Random, age: int, today: date) -> date:
    # 生日落在 (today-(age+1)年, today-age年] 之間 → 當日實歲恰為 age
    start = _shift_years(today, age + 1) + timedelta(days=1)
    end = _shift_years(today, age)
    return start + timedelta(days=rng.randint(0, (end - start).days))


def generate_patients(n: int = 20, seed: int = 42, today: date | None = None) -> list[dict]:
    rng = random.Random(seed)
    fake = Faker("zh_TW")
    fake.seed_instance(seed)
    today = today or date.today()

    patients = []
    for i in range(n):
        sex = rng.choice(["男", "女"])
        # 高齡為主:用藥品項多,欄位涵蓋較廣
        age = rng.randint(55, 92)
        birth = _birth_date_from_age(rng, age, today)
        patients.append({
            "patient_id": f"P{i + 1:03d}",
            "is_synthetic": True,
            "name": fake.name_male() if sex == "男" else fake.name_female(),
            "sex": sex,
            "age": age,
            "birth_date": birth.isoformat(),
            "national_id": _fake_national_id(rng, sex),
            "medical_record_no": "".join(rng.choices(string.digits, k=8)),
        })
    return patients


def save_patients(patients: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"is_synthetic": True, "patients": patients},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
