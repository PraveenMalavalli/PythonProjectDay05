"""Transparent per-field accuracy evaluation against human-reviewed ground truth."""

import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

FIELDS = (
    "full_name",
    "email",
    "phone",
    "location",
    "years_experience",
    "skills",
    "work_experience",
    "education",
)


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Ground truth must be a mapping of document ID to fields")
    return data


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        document_id = record["document_id"]
        if document_id in predictions:
            raise ValueError(f"Duplicate prediction ID: {document_id}")
        predictions[document_id] = record
    return predictions


def _normal(value: Any, field: str) -> Any:
    if value is None:
        return None
    if field == "phone":
        return re.sub(r"\D", "", str(value))
    if field == "skills":
        return sorted(set(str(v).strip().casefold() for v in value))
    if field == "work_experience":
        return [
            (
                job["title"].strip().casefold(),
                job["company"].strip().casefold(),
                int(job["start_year"]),
                int(job["end_year"]),
            )
            for job in value
        ]
    if field == "education":
        return (value["degree"].strip().casefold(), int(value["graduation_year"]))
    if isinstance(value, str):
        return value.strip().casefold()
    return value


def field_correct(expected: Any, actual: Any, field: str) -> bool:
    try:
        return _normal(expected, field) == _normal(actual, field)
    except (KeyError, ValueError, TypeError, AttributeError):
        return False


def evaluate(
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """All expected fields stay in denominator, even when an entire extraction fails."""
    total = len(ground_truth) * len(FIELDS)
    correct = 0
    successful = 0
    per_field = {field: {"correct": 0, "total": len(ground_truth)} for field in FIELDS}
    per_document: list[dict[str, Any]] = []
    durations: list[float] = []
    for document_id, expected in ground_truth.items():
        prediction = predictions.get(document_id, {})
        ok = prediction.get("status") == "ok" and isinstance(prediction.get("data"), dict)
        actual = prediction["data"] if ok else {}
        if ok:
            successful += 1
        if "duration_seconds" in prediction:
            durations.append(float(prediction["duration_seconds"]))
        document_correct = 0
        for field in FIELDS:
            passed = field in actual and field_correct(expected[field], actual[field], field)
            if passed:
                correct += 1
                document_correct += 1
                per_field[field]["correct"] += 1
        per_document.append(
            {
                "document_id": document_id,
                "correct": document_correct,
                "total": len(FIELDS),
                "accuracy_percent": round(100 * document_correct / len(FIELDS), 2),
                "status": prediction.get("status", "missing"),
            }
        )
    for score in per_field.values():
        score["accuracy_percent"] = round(
            100 * score["correct"] / score["total"], 2
        ) if score["total"] else None
    return {
        "documents": len(ground_truth),
        "successful_documents": successful,
        "failed_or_missing_documents": len(ground_truth) - successful,
        "correct_fields": correct,
        "total_fields": total,
        "field_accuracy_percent": round(100 * correct / total, 2) if total else None,
        "average_seconds_per_attempted_document": round(mean(durations), 4) if durations else None,
        "total_input_tokens_reported": sum(
            row.get("input_tokens") or 0 for row in predictions.values()
        ),
        "total_output_tokens_reported": sum(
            row.get("output_tokens") or 0 for row in predictions.values()
        ),
        "by_field": per_field,
        "by_document": per_document,
        "note": (
            "Skills are compared as case-insensitive sets, phone as digits, and jobs in source "
            "order. Failed/missing CVs score zero for their eight expected fields. "
            "Token counts include only usage reported by the server."
        ),
    }


def compare_models(
    ground_truth: dict[str, dict[str, Any]],
    local: dict[str, dict[str, Any]],
    hosted: dict[str, dict[str, Any]],
    output_csv: Path,
) -> dict[str, Any]:
    """Create a field-level manual-review CSV plus a genuine summary JSON."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "cv_id", "field", "ground_truth", "local_value", "local_correct",
                "hosted_value", "hosted_correct", "local_status", "hosted_status",
                "local_duration_seconds", "hosted_duration_seconds",
            ],
        )
        writer.writeheader()
        for cv_id, actual_truth in ground_truth.items():
            loc = local.get(cv_id, {})
            hos = hosted.get(cv_id, {})
            loc_data = loc.get("data") if loc.get("status") == "ok" else None
            hos_data = hos.get("data") if hos.get("status") == "ok" else None
            for field in FIELDS:
                lv = loc_data.get(field) if isinstance(loc_data, dict) else None
                hv = hos_data.get(field) if isinstance(hos_data, dict) else None
                writer.writerow(
                    {
                        "cv_id": cv_id,
                        "field": field,
                        "ground_truth": json.dumps(actual_truth[field], ensure_ascii=False),
                        "local_value": json.dumps(lv, ensure_ascii=False),
                        "local_correct": lv is not None
                        and field_correct(actual_truth[field], lv, field),
                        "hosted_value": json.dumps(hv, ensure_ascii=False),
                        "hosted_correct": hv is not None
                        and field_correct(actual_truth[field], hv, field),
                        "local_status": loc.get("status", "missing"),
                        "hosted_status": hos.get("status", "missing"),
                        "local_duration_seconds": loc.get("duration_seconds"),
                        "hosted_duration_seconds": hos.get("duration_seconds"),
                    }
                )
    summary = {"local": evaluate(ground_truth, local), "hosted": evaluate(ground_truth, hosted)}
    summary_path = output_csv.with_name(output_csv.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary
