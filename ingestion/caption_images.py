import hashlib
import json
import mimetypes
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from gemini_generation import generate


MODEL = "gemini-3.1-flash-lite"
VERSION = 1
CHUNKS_DIRECTORY = Path("chunks")
OUTPUT_PATH = CHUNKS_DIRECTORY / "image-captions" / "chunks.jsonl"
CACHE_PATH = CHUNKS_DIRECTORY / "image-captions" / "cache.jsonl"

PROMPT = """
Create a concise, retrieval-focused caption for this document image.

Return SKIP if the image is only a logo, icon, decoration, separator, product
thumbnail, or contains no useful technical information.

For a useful technical image, use this structure:

Component: Identify the board, module, document page, table, or interface.
Purpose: State what the image demonstrates.
Technical details: Preserve visible labels, values, pins, connections, table
fields, chart relationships, warnings, and protocols exactly.
Uncertainty: State anything that cannot be read confidently.

Rules:
- Treat explicit nearby document text as authoritative when visual alignment
  is ambiguous.
- Never replace documented values with guessed visual values.
- Do not infer a connection from wire color alone.
- Mark unreadable details as uncertain instead of guessing.
- Omit unrelated labels and keep the caption under 180 words.
"""


def read_chunks():
    chunks = []
    for path in CHUNKS_DIRECTORY.glob("*/chunks.jsonl"):
        if path == OUTPUT_PATH:
            continue
        with path.open(encoding="utf-8") as file:
            chunks.extend(json.loads(line) for line in file if line.strip())
    return chunks


def image_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def large_enough(path):
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return False
    return width >= 160 and height >= 100


def useful_location(image_chunk):
    location = image_chunk["metadata"]["source_location"].lower()
    skipped_sections = {"kit list"}
    return not any(section in location for section in skipped_sections)


def nearby_text(image_chunk, chunks_by_id):
    parent_id = image_chunk["metadata"].get("parent_chunk_id")
    source = image_chunk["metadata"]["source_document"]
    parent = chunks_by_id.get((source, parent_id), {})
    return parent.get("text", "")[:4000]


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    cache = {}
    with CACHE_PATH.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            cache[(record["version"], record["image_hash"])] = record["caption"]
    return cache


def save_cache_record(record):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_caption(client, image_chunk, context):
    path = Path(image_chunk["image_path"])
    image = types.Part.from_bytes(
        data=path.read_bytes(),
        mime_type=mimetypes.guess_type(path)[0],
    )
    prompt = (
        f"{PROMPT}\n\n"
        f"Source location:\n"
        f"{image_chunk['metadata']['source_location']}\n\n"
        f"Nearby document text:\n{context}"
    )
    return generate(client, MODEL, [prompt, image]).text.strip()


def write_caption_chunks(caption_chunks):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk in caption_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main():
    client = genai.Client()
    chunks = read_chunks()
    chunks_by_id = {
        (chunk["metadata"]["source_document"], chunk["chunk_id"]): chunk
        for chunk in chunks
        if chunk["modality"] == "text"
    }
    image_chunks = [
        chunk
        for chunk in chunks
        if chunk["modality"] == "image"
        and Path(chunk["image_path"]).exists()
        and large_enough(Path(chunk["image_path"]))
        and useful_location(chunk)
    ]
    cache = load_cache()
    captions_by_hash = {}

    for number, image_chunk in enumerate(image_chunks, 1):
        path = Path(image_chunk["image_path"])
        digest = image_hash(path)
        key = (VERSION, digest)

        if digest in captions_by_hash:
            continue

        if key in cache:
            caption = cache[key]
        else:
            caption = create_caption(
                client,
                image_chunk,
                nearby_text(image_chunk, chunks_by_id),
            )
            save_cache_record(
                {
                    "version": VERSION,
                    "image_hash": digest,
                    "image_path": str(path),
                    "caption": caption,
                }
            )

        captions_by_hash[digest] = {
            "caption": caption,
            "image_chunk": image_chunk,
        }
        print(f"{number}/{len(image_chunks)} {path}")

        caption_chunks = build_caption_chunks(captions_by_hash)
        write_caption_chunks(caption_chunks)

    caption_chunks = build_caption_chunks(captions_by_hash)
    write_caption_chunks(caption_chunks)
    print(f"Saved {len(caption_chunks)} useful unique captions to {OUTPUT_PATH}")


def build_caption_chunks(captions_by_hash):
    caption_chunks = []
    for digest, record in captions_by_hash.items():
        caption = record["caption"]
        if caption.strip().upper() == "SKIP":
            continue
        image_chunk = record["image_chunk"]
        caption_chunks.append(
            {
                "chunk_id": f"caption-{digest[:16]}",
                "modality": "text",
                "text": caption,
                "metadata": {
                    **image_chunk["metadata"],
                    "content_type": "image_caption",
                    "image_path": image_chunk["image_path"],
                    "image_hash": digest,
                },
            }
        )
    return caption_chunks


if __name__ == "__main__":
    main()
