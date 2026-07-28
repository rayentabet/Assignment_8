from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from embedding import index_chunks
from ingestion import ingest_document
from retrieval import search
from .adapters import load_adapter
from .dataset import load_dataset
from .metrics import calculate_metrics
from .runner import run_evaluation


app = FastAPI(title="RAG Evaluation API")
RUNS_DIRECTORY = Path("runs")
UPLOADS_DIRECTORY = Path("uploads")
INGESTED_DIRECTORY = Path("ingested")


class RunRequest(BaseModel):
    dataset_path: str
    adapter: str
    output_dir: str = "runs/latest"
    limit: int | None = None


class MetricsRequest(BaseModel):
    predictions_path: str
    output_path: str = "runs/latest/metrics.csv"


class IndexRequest(BaseModel):
    chunks_directory: str = "chunks"


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class AskRequest(BaseModel):
    query: str


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_csv(path: Path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


@app.post("/evaluate")
def evaluate_rag(request: RunRequest):
    try:
        records = load_dataset(Path(request.dataset_path))
        if request.limit:
            records = records[: request.limit]

        adapter = load_adapter(request.adapter, records)
        output_path = run_evaluation(records, adapter, Path(request.output_dir))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "number_of_questions": len(records),
        "predictions_path": str(output_path),
    }


@app.post("/metrics")
def evaluate_metrics(request: MetricsRequest):
    try:
        calculate_metrics(
            Path(request.predictions_path),
            Path(request.output_path),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    metrics = pd.read_csv(request.output_path)
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    averages = {
        name: (
            metrics[name].mean()
            if pd.notna(metrics[name].mean())
            else None
        )
        for name in metric_names
        if name in metrics.columns
    }

    return {
        "average_scores": averages,
        "metrics_path": request.output_path,
    }


@app.get("/runs")
def list_runs():
    RUNS_DIRECTORY.mkdir(exist_ok=True)
    return [
        {
            "name": directory.name,
            "has_predictions": (directory / "predictions.jsonl").exists(),
            "has_metrics": (directory / "metrics.csv").exists(),
        }
        for directory in sorted(RUNS_DIRECTORY.iterdir())
        if directory.is_dir()
    ]


@app.get("/runs/{run_name}")
def get_run(run_name: str):
    run_directory = RUNS_DIRECTORY / run_name
    predictions_path = run_directory / "predictions.jsonl"
    metrics_path = run_directory / "metrics.csv"

    return {
        "predictions": read_jsonl(predictions_path)
        if predictions_path.exists()
        else [],
        "metrics": read_csv(metrics_path) if metrics_path.exists() else [],
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    document_id = uuid.uuid4().hex[:8]
    filename = Path(file.filename).name
    upload_directory = UPLOADS_DIRECTORY / document_id
    upload_path = upload_directory / filename
    output_directory = INGESTED_DIRECTORY / document_id

    upload_directory.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(await file.read())

    report = ingest_document(upload_path, output_directory)
    if report["text_chunks"] > 0:
        report["indexing"] = index_chunks(output_directory)
    else:
        report["indexing"] = {
            "indexed_chunks": 0,
            "reason": "No text was extracted, so nothing was indexed.",
        }
    report["document_id"] = document_id
    report["original_filename"] = filename
    return report


@app.post("/index")
def index(request: IndexRequest):
    return index_chunks(Path(request.chunks_directory))


@app.post("/search")
def search_chunks(request: SearchRequest):
    return search(request.query, request.limit)


@app.post("/ask")
def ask_question(request: AskRequest):
    from dataclasses import asdict
    from rag_adapter import adapter

    response = adapter.answer(request.query, "")
    return {
        "query": request.query,
        "answer": response.answer,
        "contexts": [asdict(context) for context in response.contexts],
    }
