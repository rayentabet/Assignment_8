import json
import mimetypes
from pathlib import Path

from google import genai
from google.genai import types

from gemini_generation import generate


MODEL = "gemini-3.1-flash-lite"
DATASET_PATH = Path("arduino_rag_gold_dataset_final.jsonl")
CHUNKS_PATH = Path("chunks/KS0399,0400,0401/chunks.jsonl")
OUTPUT_PATH = Path("chunks/image-caption-pilot/chunks.jsonl")

CAPTION_PROMPT = """
Create a concise, retrieval-focused caption for this technical diagram.

Use this structure:

Component: Identify the board and module using the document context.
Purpose: State what the diagram demonstrates.
Connections:
- List every visible connection as "module pin → board pin".
- Preserve pin names and numbers exactly.
- Distinguish GND, voltage, and signal pins.
Visible labels: List only labels directly relevant to the connections.
Uncertainty: State which labels or endpoints cannot be read confidently.

Rules:
- Treat explicit pin connections in the nearby document context as
  authoritative when the dense image layout is visually ambiguous.
- Use the image to confirm the module, wires, and visible endpoints.
- Never replace a documented pin number with a guessed visual alignment.
- If the image and document appear inconsistent, report the conflict.
- Do not infer a connection from wire color alone.
- Never guess an unreadable pin. Mark it as uncertain instead.
- Omit unrelated board labels and keep the entire caption under 180 words.
"""


def load_target_filenames():
    records = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()[:15]
    ]
    return {
        Path(record["related_image_path"]).name
        for record in records
        if record["image_required"] and record["related_image_path"]
    }


def load_chunks():
    with CHUNKS_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def load_target_chunks(chunks, filenames):
    return [
        chunk
        for chunk in chunks
        if (
            chunk["modality"] == "image"
            and Path(chunk["image_path"]).name in filenames
        )
    ]


def project_context(image_chunk, all_chunks):
    location = image_chunk["metadata"]["source_location"]
    project = next(
        (
            part.strip()
            for part in location.split(">")
            if part.strip().lower().startswith("project")
        ),
        location,
    )
    useful_sections = {
        "specification",
        "hardware connection",
        "components introduction",
    }
    context_chunks = []
    for chunk in all_chunks:
        chunk_location = chunk.get("metadata", {}).get("source_location", "")
        if chunk.get("modality") != "text" or project not in chunk_location:
            continue
        if any(section in chunk_location.lower() for section in useful_sections):
            context_chunks.append(chunk["text"])
    return "\n\n".join(context_chunks)[:4000]


def caption_image(client, image_chunk, context):
    image_path = image_chunk["image_path"]
    path = Path(image_path)
    image = types.Part.from_bytes(
        data=path.read_bytes(),
        mime_type=mimetypes.guess_type(path)[0],
    )
    prompt = (
        f"{CAPTION_PROMPT}\n\n"
        f"Source location:\n"
        f"{image_chunk['metadata']['source_location']}\n\n"
        f"Nearby document context:\n{context}"
    )
    response = generate(client, MODEL, [prompt, image])
    return response.text.strip()


def main():
    client = genai.Client()
    chunks = load_chunks()
    image_chunks = load_target_chunks(chunks, load_target_filenames())
    caption_chunks = []

    for image_chunk in image_chunks:
        caption = caption_image(
            client,
            image_chunk,
            project_context(image_chunk, chunks),
        )
        caption_chunks.append(
            {
                "chunk_id": f"caption-{image_chunk['chunk_id']}",
                "modality": "text",
                "text": caption,
                "metadata": {
                    **image_chunk["metadata"],
                    "content_type": "image_caption",
                    "image_path": image_chunk["image_path"],
                },
            }
        )
        print(image_chunk["image_path"])
        print(caption)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk in caption_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved {len(caption_chunks)} captions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
