import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldRecord:
    id: str
    query: str
    expected_retrieved_information: str
    expected_answer: str
    source_document: str
    source_location: str
    evidence_modality: str
    image_required: bool
    related_image_path: str = ""
    answer_checklist: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "GoldRecord":
        return cls(
            id=data["id"],
            query=data["query"],
            expected_retrieved_information=data["expected_retrieved_information"],
            expected_answer=data["expected_answer"],
            source_document=data["source_document"],
            source_location=data["source_location"],
            evidence_modality=data["evidence_modality"],
            image_required=data["image_required"],
            related_image_path=data.get("related_image_path", ""),
            answer_checklist=data.get("answer_checklist", ""),
        )


def load_dataset(path: Path) -> list[GoldRecord]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(GoldRecord.from_dict(json.loads(line)))
    return records

