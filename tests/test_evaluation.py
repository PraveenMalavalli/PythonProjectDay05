import csv
import json

from app.services.evaluation import (
    FIELDS,
    compare_models,
    evaluate,
    field_correct,
    load_ground_truth,
    load_predictions,
)
from tests.conftest import ROOT


def test_ground_truth_has_10_cv_records_and_eight_fields():
    truth = load_ground_truth(ROOT / "ground_truth/cvs_01_10.json")
    assert len(truth) == 10
    assert all(set(row) == set(FIELDS) for row in truth.values())


def test_field_comparison_normalizes_phone_and_skills():
    assert field_correct("+00 555 588 8287", "005555888287", "phone")
    assert field_correct(["Python", "SQL"], ["sql", "python"], "skills")
    assert not field_correct(["Python", "SQL"], ["Python"], "skills")


def test_successful_one_cv_scores_eight_of_eighty():
    truth = load_ground_truth(ROOT / "ground_truth/cvs_01_10.json")
    predictions = {"cv_01": {"document_id": "cv_01", "status": "ok",
                             "data": truth["cv_01"], "duration_seconds": 0.5}}
    score = evaluate(truth, predictions)
    assert score["correct_fields"] == 8
    assert score["total_fields"] == 80
    assert score["field_accuracy_percent"] == 10.0
    assert score["failed_or_missing_documents"] == 9


def test_missing_or_failed_predictions_score_zero():
    truth = load_ground_truth(ROOT / "ground_truth/cvs_01_10.json")
    predictions = {"cv_01": {"document_id": "cv_01", "status": "failed",
                             "data": truth["cv_01"]}}
    score = evaluate(truth, predictions)
    assert score["correct_fields"] == 0
    assert score["failed_or_missing_documents"] == 10


def test_compare_csv_has_80_fields_and_real_results(tmp_path):
    truth = load_ground_truth(ROOT / "ground_truth/cvs_01_10.json")
    local = {cv_id: {"document_id": cv_id, "status": "ok", "data": data,
                     "duration_seconds": 2.0} for cv_id, data in truth.items()}
    hosted = {}  # no run: must score as missing, not assume success
    path = tmp_path / "comparison.csv"
    summary = compare_models(truth, local, hosted, path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 80
    assert summary["local"]["field_accuracy_percent"] == 100.0
    assert summary["hosted"]["field_accuracy_percent"] == 0.0
    assert path.with_name("comparison_summary.json").exists()


def test_prediction_jsonl_loader_rejects_duplicate_cv_ids(tmp_path):
    p = tmp_path / "duplicates.jsonl"
    line = json.dumps({"document_id": "cv_01", "status": "failed"})
    p.write_text(line + "\n" + line + "\n")
    try:
        load_predictions(p)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Duplicate document IDs should be rejected")
