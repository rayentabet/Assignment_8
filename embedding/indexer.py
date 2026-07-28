import json
import os
import uuid
from pathlib import Path

from google import genai
from qdrant_client import QdrantClient, models

from .embeddings import MODEL, VECTOR_SIZE, embed_batch_with_retry


def load_chunks(chunks_directory: Path):
    chunks = []
    paths = list(chunks_directory.glob("*/chunks.jsonl"))
    direct_path = chunks_directory / "chunks.jsonl"
    if direct_path.exists():
        paths.append(direct_path)

    for path in paths:
        with path.open(encoding="utf-8") as file:
            chunks.extend(json.loads(line) for line in file if line.strip())
    return chunks


def chunk_payload(chunk):
    payload = {
        "chunk_id": chunk["chunk_id"],
        "modality": chunk["modality"],
        **chunk["metadata"],
    }
    if chunk["modality"] == "text":
        payload["text"] = chunk["text"]
    else:
        payload["image_path"] = chunk["image_path"]
    return payload


def index_chunks(chunks_directory: Path):
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        check_compatibility=False,
    )
    collection = os.environ["QDRANT_COLLECTION"]

    if not qdrant.collection_exists(collection):
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    chunks = load_chunks(chunks_directory)
    stored_points, _ = qdrant.scroll(
        collection_name=collection,
        limit=10000,
        with_payload=False,
        with_vectors=False,
    )
    stored_ids = {str(point.id) for point in stored_points}

    text_chunks = [chunk for chunk in chunks if chunk["modality"] == "text"]
    indexed = 0
    skipped = 0

    for group, batch_size in [(text_chunks, 20)]:
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            new_chunks = []

            for chunk in batch:
                source = chunk["metadata"]["source_document"]
                point_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, source + chunk["chunk_id"])
                )
                if point_id in stored_ids:
                    skipped += 1
                else:
                    new_chunks.append(chunk)

            if not new_chunks:
                continue

            vectors = embed_batch_with_retry(gemini, new_chunks)
            points = []

            for chunk, vector in zip(new_chunks, vectors):
                source = chunk["metadata"]["source_document"]
                point_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, source + chunk["chunk_id"])
                )
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=chunk_payload(chunk),
                    )
                )

            qdrant.upsert(collection_name=collection, points=points)
            indexed += len(points)
            print(f"Indexed {indexed + skipped}/{len(chunks)} chunks")

    return {
        "collection": collection,
        "indexed_chunks": indexed,
        "skipped_chunks": skipped,
        "vector_size": VECTOR_SIZE,
        "model": MODEL,
    }
