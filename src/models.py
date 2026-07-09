"""Label JSON 的 pydantic schema(spec §7.2)。

所有輸出標註一律經過這裡驗證再落地;
validate 指令會把 output/labels/*.json 全部 round-trip 一次。
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BBox(BaseModel):
    """影像座標(像素),左上角原點。"""
    x: int
    y: int
    w: int
    h: int

    def corners(self) -> list[tuple[float, float]]:
        return [(self.x, self.y), (self.x + self.w, self.y),
                (self.x + self.w, self.y + self.h), (self.x, self.y + self.h)]


PIIField = Literal["patient_name", "patient_id", "national_id",
                   "birth_date", "address", "phone", "medical_record_no"]


class PIIRegion(BaseModel):
    field: PIIField
    bbox: BBox
    text: str                # 合成的假值(供還原比對)
    redact: bool = True      # 是否需去識別化遮蔽


class OCRLine(BaseModel):
    text: str
    bbox: BBox


class PrescriptionItem(BaseModel):
    index: int
    drug_name_en: str
    drug_name_zh: str
    strength: str            # "5mg"
    dosage_form: str         # "錠"
    dose_per_admin: float    # 每次幾顆
    unit: str                # "顆"/"mL"/"下"/"單位"
    frequency_code: str      # "BID"
    frequency_text: str      # "每日兩次"
    timing_code: str         # "PC"
    timing_text: str         # "飯後"
    duration_days: int
    total_qty: float
    prn: bool
    indication: str
    field_bboxes: dict[str, BBox]  # {"drug_name":.., "frequency":.., "timing":.., "dose":..}


class Sample(BaseModel):
    sample_id: str                       # "twmedbag_00042"
    image: str                           # 相對路徑
    image_size: tuple[int, int]          # (width, height)
    hospital_template: str
    is_synthetic: Literal[True] = True
    patient_ref: str                     # 模擬病人 id(對到 patients.json)
    pii_regions: list[PIIRegion]
    prescription: PrescriptionItem       # 一袋一藥
    ocr_lines: list[OCRLine]             # 供純 OCR 訓練
    augmentation: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def bboxes_inside_image(self) -> "Sample":
        w, h = self.image_size
        boxes = [r.bbox for r in self.pii_regions]
        boxes += list(self.prescription.field_bboxes.values())
        boxes += [ln.bbox for ln in self.ocr_lines]
        for b in boxes:
            if b.x < 0 or b.y < 0 or b.x + b.w > w or b.y + b.h > h:
                raise ValueError(
                    f"{self.sample_id}: bbox {b} 超出影像範圍 {self.image_size}")
        return self
