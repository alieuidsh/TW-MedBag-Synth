"""把渲染結果 + 處方 ground truth 組成 label JSON(經 pydantic 驗證)。"""
import json
from pathlib import Path

from config.enums import FREQUENCY, TIMING
from src.models import BBox, OCRLine, PIIRegion, PrescriptionItem, Sample

# data-field 名稱 → PIIRegion.field(spec §7.2 Literal)
_PII_FIELD_MAP = {
    "patient_name": "patient_name",
    "medical_record_no": "medical_record_no",
    "national_id": "national_id",
    "birth_date": "birth_date",
}


def build_sample(sample_id: str, image_rel: str, image_size: tuple[int, int],
                 template_id: str, rx: dict,
                 fields: list[dict], lines: list[dict],
                 augmentation: list[str]) -> Sample:
    drug = rx["drug"]

    pii_regions = [
        PIIRegion(field=_PII_FIELD_MAP[f["field"]], bbox=BBox(**f["bbox"]),
                  text=f["text"], redact=True)
        for f in fields if f["pii"] and f["field"] in _PII_FIELD_MAP
    ]
    field_bboxes = {f["field"]: BBox(**f["bbox"]) for f in fields if not f["pii"]}

    item = PrescriptionItem(
        index=rx["index"],
        drug_name_en=drug["name_en"],
        drug_name_zh=drug["name_zh"],
        strength=rx["strength"],
        dosage_form=drug["form"],
        dose_per_admin=rx["dose_per_admin"],
        unit=rx["unit"],
        frequency_code=drug["typical_freq"],
        frequency_text=FREQUENCY[drug["typical_freq"]][0],
        timing_code=drug["typical_timing"],
        timing_text=TIMING[drug["typical_timing"]],
        duration_days=rx["duration_days"],
        total_qty=rx["total_qty"],
        prn=rx["prn"],
        indication=drug["indication"],
        field_bboxes=field_bboxes,
    )

    return Sample(
        sample_id=sample_id,
        image=image_rel,
        image_size=image_size,
        hospital_template=template_id,
        patient_ref=rx["patient"]["patient_id"],
        pii_regions=pii_regions,
        prescription=item,
        ocr_lines=[OCRLine(text=ln["text"], bbox=BBox(**ln["bbox"]))
                   for ln in lines if ln["text"]],
        augmentation=augmentation,
    )


def write_label(sample: Sample, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    path = labels_dir / f"{sample.sample_id}.json"
    path.write_text(sample.model_dump_json(indent=2), encoding="utf-8")
