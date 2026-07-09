"""schema 與 formulary 的基本驗證(pytest)。"""
from pathlib import Path

import pytest
import yaml

from config.enums import FORM_UNIT, FREQUENCY, TIMING
from src.models import BBox, OCRLine, PIIRegion, PrescriptionItem, Sample
from src.prescriptions import appearance_text

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def drugs() -> list[dict]:
    data = yaml.safe_load((ROOT / "config" / "drugs.yaml").read_text(encoding="utf-8"))
    return data["drugs"]


def test_drugs_exactly_100(drugs):
    assert len(drugs) == 100


def test_drugs_unique_names(drugs):
    names = [d["name_en"] for d in drugs]
    assert len(set(names)) == 100


def test_drugs_fields_complete(drugs):
    required = {"name_en", "name_zh", "strengths", "form",
                "typical_freq", "typical_timing", "indication"}
    for d in drugs:
        assert required <= set(d), f"{d.get('name_en')} 缺欄位"
        assert d["typical_freq"] in FREQUENCY, d["name_en"]
        assert d["typical_timing"] in TIMING, d["name_en"]
        assert d["form"] in FORM_UNIT, d["name_en"]
        assert len(d["strengths"]) >= 1


def test_appearance_deterministic(drugs):
    # 同一種藥每次執行外觀必須一致(crc32,非加鹽 hash)
    assert appearance_text(drugs[0]) == appearance_text(drugs[0])


def _make_item(**field_bboxes) -> PrescriptionItem:
    return PrescriptionItem(
        index=0, drug_name_en="Amlodipine", drug_name_zh="脈優",
        strength="5mg", dosage_form="錠", dose_per_admin=1, unit="顆",
        frequency_code="QD", frequency_text="每日一次",
        timing_code="PC", timing_text="飯後",
        duration_days=28, total_qty=28, prn=False, indication="降血壓",
        field_bboxes=field_bboxes)


def test_sample_valid():
    s = Sample(
        sample_id="twmedbag_00000", image="images/twmedbag_00000.png",
        image_size=(640, 880), hospital_template="generic_v1",
        patient_ref="P001",
        pii_regions=[PIIRegion(field="patient_name",
                               bbox=BBox(x=10, y=10, w=60, h=20), text="王大明")],
        prescription=_make_item(drug_name=BBox(x=10, y=100, w=300, h=24)),
        ocr_lines=[OCRLine(text="藥品名稱:脈優", bbox=BBox(x=8, y=98, w=320, h=26))])
    assert s.is_synthetic is True


def test_bbox_outside_image_rejected():
    with pytest.raises(ValueError):
        Sample(
            sample_id="twmedbag_00001", image="x.png",
            image_size=(640, 880), hospital_template="generic_v1",
            patient_ref="P001", pii_regions=[],
            prescription=_make_item(drug_name=BBox(x=600, y=100, w=100, h=24)),
            ocr_lines=[])
