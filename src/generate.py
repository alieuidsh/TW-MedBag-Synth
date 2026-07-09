"""主流程 CLI(typer)。

    python -m src.generate patients --n 20 --out output/patients.json
    python -m src.generate build --aug-per-sample 3 --out output
    python -m src.generate deid --in output --out output/redacted
    python -m src.generate validate --in output
"""
import json
import random
from pathlib import Path

import typer
import yaml
from faker import Faker
from PIL import Image

from src import augment as aug
from src.deid import redact_dataset
from src.labeler import build_sample, write_label
from src.models import Sample
from src.patients import generate_patients, save_patients
from src.prescriptions import age_at, load_drugs, make_prescriptions
from src.render import Renderer, drug_display_name, sample_variation

app = typer.Typer(help="TW-MedBag-Synth 合成藥袋資料集生成器", add_completion=False)


@app.command()
def patients(n: int = typer.Option(20, help="模擬病人數"),
             seed: int = 42,
             out: Path = Path("output/patients.json")) -> None:
    """生成模擬病人(全假值)。"""
    data = generate_patients(n=n, seed=seed)
    save_patients(data, out)
    typer.echo(f"已生成 {len(data)} 位模擬病人 → {out}")


def _transform_annotations(fields: list[dict], lines: list[dict],
                           H, size: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    if H is None:
        return fields, lines
    w, h = size
    tf = [{**f, "bbox": aug.transform_bbox(f["bbox"], H, w, h)} for f in fields]
    tl = [{**ln, "bbox": aug.transform_bbox(ln["bbox"], H, w, h)} for ln in lines]
    return tf, tl


@app.command()
def build(patients_file: Path = typer.Option(Path("output/patients.json"), "--patients"),
          drugs_file: Path = typer.Option(Path("config/drugs.yaml"), "--drugs"),
          templates_file: Path = typer.Option(Path("config/templates.yaml"), "--templates"),
          aug_per_sample: int = typer.Option(3, help="每張乾淨藥袋另產生幾個增強變體"),
          seed: int = 42,
          out: Path = Path("output")) -> None:
    """處方組合 → 渲染 → 標註 → augmentation。"""
    rng = random.Random(seed)
    fake = Faker("zh_TW")
    fake.seed_instance(seed + 1)

    patient_list = json.loads(patients_file.read_text(encoding="utf-8"))["patients"]
    drugs = load_drugs(drugs_file)
    tcfg = yaml.safe_load(templates_file.read_text(encoding="utf-8"))
    templates = tcfg["templates"]
    weights = [t["weight"] for t in templates]

    rxs = make_prescriptions(patient_list, drugs, seed=seed)
    typer.echo(f"病人 {len(patient_list)} 位 × 藥物 {len(drugs)} 種 → 處方 {len(rxs)} 筆")

    images_dir = out / "images"
    labels_dir = out / "labels"
    counter = 0
    sample_index: dict[str, list[str]] = {p["patient_id"]: [] for p in patient_list}

    with Renderer("templates") as renderer:
        for i, rx in enumerate(rxs):
            template = rng.choices(templates, weights=weights, k=1)[0]
            variation = sample_variation(tcfg["variation"], rng)
            ctx = {
                **variation,
                "hospital_name": rng.choice(tcfg["hospitals"][template["id"]]),
                "dispense_date": rx["dispense_date"],
                "patient": {**rx["patient"],
                            "age": age_at(rx["patient"]["birth_date"],
                                          rx["dispense_date"])},
                "drug_display": drug_display_name(rx, variation["name_order"]),
                "dosage_parts": rx["dosage_parts"],
                "total_text": rx["total_text"],
                "indication": rx["drug"]["indication"],
                "appearance_text": rx["appearance_text"],
                "warning_text": rx["warning_text"],
                "doctor_name": fake.name(),
                "pharmacist_name": fake.name(),
            }

            # 1) 乾淨版
            sid = f"twmedbag_{counter:05d}"
            png = images_dir / f"{sid}.png"
            size, fields, lines = renderer.render(template, ctx, png)
            sample = build_sample(sid, f"images/{sid}.png", size,
                                  template["id"], rx, fields, lines, [])
            write_label(sample, labels_dir)
            sample_index[rx["patient"]["patient_id"]].append(sid)
            counter += 1

            # 2) 增強變體(幾何類需同步變換 bbox)
            base_img = aug.imread_u(png)
            for _ in range(aug_per_sample):
                sid = f"twmedbag_{counter:05d}"
                img_v, H, names = aug.make_variant(base_img, rng)
                aug.imwrite_u(images_dir / f"{sid}.png", img_v)
                f_v, l_v = _transform_annotations(fields, lines, H, size)
                sample = build_sample(sid, f"images/{sid}.png", size,
                                      template["id"], rx, f_v, l_v, names)
                write_label(sample, labels_dir)
                sample_index[rx["patient"]["patient_id"]].append(sid)
                counter += 1

            if (i + 1) % 25 == 0:
                typer.echo(f"  進度 {i + 1}/{len(rxs)} 筆處方(累計 {counter} 張)")

    # split 依「病人」切,同一病人不跨 train/val/test(spec §9)
    pids = [p["patient_id"] for p in patient_list]
    rng.shuffle(pids)
    n_train = round(len(pids) * 0.7)
    n_val = round(len(pids) * 0.15)
    split_pids = {"train": pids[:n_train],
                  "val": pids[n_train:n_train + n_val],
                  "test": pids[n_train + n_val:]}
    splits = {k: sorted(s for pid in v for s in sample_index[pid])
              for k, v in split_pids.items()}

    manifest = {
        "name": "TW-MedBag-Synth",
        "version": "0.1.0",
        "homage": "Synthetic Medical Prescription OCR Dataset (Kaggle) 的台灣在地化衍生研究",
        "license": "CC BY 4.0(合成資料,無真實個資)",
        "num_patients": len(patient_list),
        "num_drugs": len(drugs),
        "num_samples": counter,
        "patient_splits": split_pids,
        "splits": splits,
        "schema_version": "1.0",
    }
    (out / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"完成:{counter} 張影像 + 標註 → {out}(manifest: dataset.json)")


@app.command()
def deid(in_dir: Path = typer.Option(Path("output"), "--in"),
         out: Path = Path("output/redacted"),
         mode: str = typer.Option("black", help="black 或 blur")) -> None:
    """對 PII 區域產生塗黑/模糊副本。"""
    n = redact_dataset(in_dir, out, mode=mode)
    typer.echo(f"已輸出 {n} 張去識別化影像 → {out}")


@app.command()
def validate(in_dir: Path = typer.Option(Path("output"), "--in")) -> None:
    """全部 label 過 pydantic schema + bbox 落在畫面內 + 影像尺寸相符。"""
    labels = sorted((in_dir / "labels").glob("*.json"))
    errors = 0
    for path in labels:
        try:
            sample = Sample.model_validate_json(path.read_text(encoding="utf-8"))
            with Image.open(in_dir / sample.image) as im:
                if im.size != tuple(sample.image_size):
                    raise ValueError(
                        f"影像尺寸 {im.size} != 標註 {sample.image_size}")
        except Exception as e:  # noqa: BLE001 — 驗證工具需回報所有錯誤型別
            typer.echo(f"[FAIL] {path.name}: {e}")
            errors += 1
    typer.echo(f"驗證 {len(labels)} 筆 label:{len(labels) - errors} 通過, {errors} 失敗")
    if errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
