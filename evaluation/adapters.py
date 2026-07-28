import importlib

from .contracts import RAGResponse, RetrievedContext
from .dataset import GoldRecord


class OracleAdapter:
    def __init__(self, records: list[GoldRecord]):
        self.records = {record.id: record for record in records}

    def answer(self, question: str, record_id: str) -> RAGResponse:
        record = self.records[record_id]
        context = RetrievedContext(
            text=record.expected_retrieved_information,
            source_id=record.source_document,
            location=record.source_location,
            image_path=record.related_image_path,
        )
        return RAGResponse(answer=record.expected_answer, contexts=[context])


def load_adapter(name: str, records: list[GoldRecord]):
    if name == "oracle":
        return OracleAdapter(records)

    module_name, object_name = name.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, object_name)

