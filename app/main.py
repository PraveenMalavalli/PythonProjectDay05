"""CLI: local/hosted CV extraction, invoice vision, evaluation and experiments."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings
from app.exceptions import LLMClientError
from app.services.document_reader import read_cv
from app.services.evaluation import (
    compare_models,
    evaluate,
    load_ground_truth,
    load_predictions,
)
from app.services.experiments import run_context_experiment, run_temperature_experiment
from app.services.extractor import StructuredExtractor
from app.services.llm_client import LLMClient, OllamaNativeClient

ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "day5_samples" / "cvs"
GROUND_TRUTH = ROOT / "ground_truth" / "cvs_01_10.json"
RESULTS_DIR = ROOT / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 5 Ollama LLM and CV extraction exercises")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check Ollama connection and installed models")

    single = commands.add_parser("extract-cv", help="Extract one text CV")
    single.add_argument("--file", type=Path, default=CV_DIR / "cv_01.txt")
    _add_generation_flags(single)

    batch = commands.add_parser("extract-all", help="Extract CVs and save JSONL results")
    batch.add_argument("--input-dir", type=Path, default=CV_DIR)
    batch.add_argument("--output", type=Path, default=RESULTS_DIR / "local_20.jsonl")
    batch.add_argument("--limit", type=int, default=20)
    _add_generation_flags(batch)

    image = commands.add_parser("extract-invoice", help="Extract invoice image or scanned PDF")
    image.add_argument("--file", type=Path, required=True)
    image.add_argument("--page", type=int, default=0, help="PDF page index, starting at 0")
    _add_generation_flags(image)

    accuracy = commands.add_parser("evaluate", help="Score 10 CVs against hand-checked reference")
    accuracy.add_argument("--predictions", type=Path, required=True)
    accuracy.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH)
    accuracy.add_argument("--output", type=Path, default=RESULTS_DIR / "local_accuracy.json")

    compare = commands.add_parser("compare", help="Hosted-versus-local per-field comparison CSV")
    compare.add_argument("--local", type=Path, required=True)
    compare.add_argument("--hosted", type=Path, required=True)
    compare.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH)
    compare.add_argument("--output", type=Path, default=RESULTS_DIR / "comparison.csv")

    temp = commands.add_parser("temperature", help="3 zero-temp, 3 high-temp, 1 fixed-seed")
    temp.add_argument("--file", type=Path, default=CV_DIR / "cv_01.txt")
    temp.add_argument("--output", type=Path, default=RESULTS_DIR / "temperature.json")
    temp.add_argument("--high-temp", type=float, default=1.2)
    temp.add_argument("--seed", type=int, default=42)
    temp.add_argument("--model", type=str, default=None)

    context = commands.add_parser("context", help="Ollama native num_ctx long-document experiment")
    context.add_argument("--file", type=Path, default=CV_DIR / "cv_01.txt")
    context.add_argument("--num-ctx", type=int, default=8192)
    context.add_argument("--repeats", type=int, default=420)
    context.add_argument("--model", type=str, default=None)
    context.add_argument("--output", type=Path, default=RESULTS_DIR / "context.json")
    return parser


def _add_generation_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["local", "hosted"], default="local")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _doctor(settings: Settings) -> int:
    base = settings.ollama_base_url.rstrip("/")
    if not base.endswith("/v1"):
        raise ValueError("OLLAMA_BASE_URL should end in /v1")
    with httpx.Client(timeout=5) as client:
        try:
            response = client.get(base[:-3] + "/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"Ollama not reachable: {exc}", file=sys.stderr)
            print("Start it with: ollama serve", file=sys.stderr)
            return 1
    names = [model.get("name") for model in response.json().get("models", [])]
    installed = settings.ollama_model in names
    print(json.dumps({"ollama_reachable": True, "installed_models": names,
                      "selected_model": settings.ollama_model,
                      "selected_model_installed": installed}, indent=2))
    if not installed:
        print(f"Pull the model: ollama pull {settings.ollama_model}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    try:
        if args.command == "doctor":
            return _doctor(settings)
        if args.command == "evaluate":
            truth = load_ground_truth(args.ground_truth)
            result = evaluate(truth, load_predictions(args.predictions))
            _write_json(args.output, result)
            print(json.dumps({"output": str(args.output), "summary": {
                "field_accuracy_percent": result["field_accuracy_percent"],
                "failed_or_missing_documents": result["failed_or_missing_documents"],
            }}, indent=2))
            return 0
        if args.command == "compare":
            summary = compare_models(
                load_ground_truth(args.ground_truth), load_predictions(args.local),
                load_predictions(args.hosted), args.output,
            )
            print(json.dumps({"csv": str(args.output), "local_accuracy":
                summary["local"]["field_accuracy_percent"], "hosted_accuracy":
                summary["hosted"]["field_accuracy_percent"]}, indent=2))
            return 0
        if args.command == "context":
            with OllamaNativeClient(
                settings.ollama_base_url, timeout=settings.llm_timeout_seconds
            ) as native:
                records = run_context_experiment(
                    native,
                    model=args.model or settings.ollama_model,
                    cv_text=read_cv(args.file), num_ctx=args.num_ctx,
                    repeats=args.repeats,
                )
            _write_json(args.output, records)
            print(json.dumps({"output": str(args.output), "runs": len(records)}, indent=2))
            return 0

        opts = settings.client_options(
            provider="local" if args.command == "temperature" else args.provider,
            model=args.model,
        )
        with LLMClient(opts) as client:
            extractor = StructuredExtractor(
                client, provider="local" if args.command == "temperature" else args.provider
            )
            if args.command == "extract-cv":
                result = extractor.extract_cv(
                    read_cv(args.file), document_id=args.file.stem,
                    prompt_version=args.prompt_version,
                    temperature=args.temperature, seed=args.seed,
                )
                print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
                return 0 if result.status == "ok" else 1

            if args.command == "extract-all":
                if not 1 <= args.limit <= 20:
                    raise ValueError("--limit must be between 1 and 20")
                paths = sorted(args.input_dir.glob("cv_*.txt"))[:args.limit]
                if not paths:
                    raise ValueError("No CV text files found")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                failures = 0
                with args.output.open("w", encoding="utf-8") as output:
                    for path in paths:
                        extracted = extractor.extract_cv(
                            read_cv(path), document_id=path.stem,
                            prompt_version=args.prompt_version,
                            temperature=args.temperature, seed=args.seed,
                        )
                        output.write(json.dumps(extracted.as_dict(), ensure_ascii=False) + "\n")
                        print(
                            f"{path.stem}: {extracted.status} "
                            f"({extracted.duration_seconds:.2f}s) {extracted.model or ''}",
                            file=sys.stderr,
                        )
                        if extracted.status != "ok":
                            failures += 1
                print(json.dumps({"output": str(args.output), "cv_count": len(paths),
                                  "failures": failures}, indent=2))
                return 0 if failures == 0 else 1

            if args.command == "extract-invoice":
                extracted = extractor.extract_invoice(
                    args.file, page=args.page, temperature=args.temperature, seed=args.seed
                )
                print(json.dumps(extracted.as_dict(), indent=2, ensure_ascii=False))
                return 0 if extracted.status == "ok" else 1

            if args.command == "temperature":
                rows = run_temperature_experiment(
                    extractor, args.file, high_temperature=args.high_temp, seed=args.seed
                )
                _write_json(args.output, rows)
                print(json.dumps({"output": str(args.output), "runs": len(rows),
                                  "successful": sum(row["status"] == "ok" for row in rows)},
                                 indent=2))
                return 0 if all(row["status"] == "ok" for row in rows) else 1

    except (ValueError, OSError, LLMClientError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
