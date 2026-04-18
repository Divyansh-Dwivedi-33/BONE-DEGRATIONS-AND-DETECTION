import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


VALID_SEX_VALUES = {"Male", "Female", "Other"}


@dataclass
class PatientPayload:
    name: str
    age: int
    sex: str
    weight: float
    notes: str


def validate_patient_payload(form) -> PatientPayload:
    name = (form.get("name") or "").strip()
    age_raw = (form.get("age") or "").strip()
    sex = (form.get("sex") or "").strip()
    weight_raw = (form.get("weight") or "").strip()
    notes = (form.get("notes") or "").strip()

    if not name:
        raise ValueError("Patient name is required.")

    try:
        age = int(age_raw)
    except (TypeError, ValueError):
        raise ValueError("Age must be a whole number.")

    if age < 1 or age > 120:
        raise ValueError("Age must be between 1 and 120.")

    if sex not in VALID_SEX_VALUES:
        raise ValueError("Sex must be one of: Male, Female, Other.")

    try:
        weight = float(weight_raw)
    except (TypeError, ValueError):
        raise ValueError("Weight must be a number.")

    if weight <= 0 or weight > 300:
        raise ValueError("Weight must be between 0 and 300 kg.")

    return PatientPayload(
        name=name,
        age=age,
        sex=sex,
        weight=weight,
        notes=notes,
    )


def save_upload(image_file: FileStorage, upload_dir: Path, allowed_extensions) -> dict:
    if not image_file.filename:
        raise ValueError("No image file selected.")

    suffix = Path(image_file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise ValueError(f"Unsupported image type. Allowed extensions: {allowed}.")

    safe_name = secure_filename(image_file.filename)
    stored_name = f"{uuid4().hex}{suffix}"
    upload_path = upload_dir / stored_name
    image_file.save(upload_path)

    return {
        "path": str(upload_path),
        "stored_filename": stored_name,
        "original_filename": safe_name,
        "content_type": image_file.content_type,
    }


def remove_upload(upload_path: Path) -> None:
    if upload_path.exists():
        upload_path.unlink()


def build_analysis_report(patient: PatientPayload, upload_meta: dict, prediction: dict) -> dict:
    analysis_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    report = {
        "analysis_id": analysis_id,
        "created_at": created_at,
        "patient": asdict(patient),
        "image": {
            "original_filename": upload_meta["original_filename"],
            "content_type": upload_meta.get("content_type"),
        },
        "prediction": {
            "label": prediction["label"],
            "title": prediction["title"],
            "description": prediction["description"],
            "recommendation": prediction["recommendation"],
            "severity": prediction["severity"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
        },
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "probs": prediction["probs"],
        "info": prediction["description"],
        "recommendation": prediction["recommendation"],
    }
    return report


def save_report(report: dict, report_dir: Path) -> Path:
    report_path = report_dir / f"{report['analysis_id']}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def get_report(report_dir: Path, analysis_id: str):
    report_path = report_dir / f"{analysis_id}.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def list_reports(report_dir: Path, limit: int = 20) -> list[dict]:
    items = []
    for report_file in sorted(report_dir.glob("*.json"), reverse=True):
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        items.append(
            {
                "analysis_id": report.get("analysis_id"),
                "created_at": report.get("created_at"),
                "patient": report.get("patient", {}),
                "label": report.get("label"),
                "confidence": report.get("confidence"),
            }
        )
        if len(items) >= max(limit, 1):
            break

    return items
