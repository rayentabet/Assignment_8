import json
import time
from dataclasses import asdict
from pathlib import Path

from .dataset import GoldRecord


def run_evaluation(records: list[GoldRecord], adapter, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.csv"

    if metrics_path.exists():
        metrics_path.unlink()

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            start_time = time.perf_counter()
            response = adapter.answer(record.query, record.id)
            latency_ms = (time.perf_counter() - start_time) * 1000

            result = {
                "id": record.id,
                "query": record.query,
                "expected_answer": record.expected_answer,
                "expected_retrieved_information": record.expected_retrieved_information,
                "source_document": record.source_document,
                "source_location": record.source_location,
                "image_required": record.image_required,
                "related_image_path": record.related_image_path,
                "answer": response.answer,
                "contexts": [asdict(context) for context in response.contexts],
                "latency_ms": round(latency_ms, 3),
            }

            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    return output_path
