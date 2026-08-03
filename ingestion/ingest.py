import hashlib
import json
import re
import zipfile
from pathlib import Path

from google import genai

from .caption_images import create_caption
from .pdf_parser import parse_pdf as parse_pdf_sections


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
BOLD_SECTION_PATTERN = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def write_chunks(chunks, output_path):
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def text_chunk(chunk_id, text, source, location, page=None):
    chunk = {
        "chunk_id": chunk_id,
        "modality": "text",
        "text": text,
        "metadata": {
            "source_document": source,
            "source_location": location,
        },
    }
    if page:
        chunk["metadata"]["page_number"] = page
    return chunk


def image_chunk(chunk_id, path, source, location, parent_id, page=None):
    chunk = {
        "chunk_id": chunk_id,
        "modality": "image",
        "image_path": str(path),
        "metadata": {
            "source_document": source,
            "source_location": location,
            "parent_chunk_id": parent_id,
        },
    }
    if page:
        chunk["metadata"]["page_number"] = page
    return chunk


def parse_pdf(path, output_directory):
    return parse_pdf_sections(path, output_directory, text_chunk, image_chunk)


def parse_markdown(path, output_directory):
    chunks = []
    headings = []
    section_title = ""
    section_lines = []
    text_number = 0
    image_number = 0

    def save_section():
        nonlocal text_number, image_number

        text = "\n".join(section_lines).strip()
        if not text:
            return

        text_number += 1
        text_id = f"text-{text_number:04d}"
        image_paths = IMAGE_PATTERN.findall(text)

        chunks.append(
            {
                "chunk_id": text_id,
                "modality": "text",
                "text": IMAGE_PATTERN.sub("", text).strip(),
                "metadata": {
                    "source_document": path.name,
                    "source_location": section_title,
                },
            }
        )

        for image_path in image_paths:
            image_number += 1
            chunks.append(
                {
                    "chunk_id": f"image-{image_number:04d}",
                    "modality": "image",
                    "image_path": str(path.parent / image_path.lstrip("/")),
                    "metadata": {
                        "source_document": path.name,
                        "source_location": section_title,
                        "parent_chunk_id": text_id,
                    },
                }
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        heading = HEADING_PATTERN.match(line)
        bold_section = BOLD_SECTION_PATTERN.match(line)

        if heading:
            save_section()
            section_lines.clear()

            level = len(heading.group(1))
            headings = headings[: level - 1]
            headings.append(heading.group(2).strip())
            section_title = " > ".join(headings)
            section_lines.append(line)

        elif (
            bold_section
            and len(bold_section.group(1).strip()) <= 60
            and not bold_section.group(1).strip().startswith("(")
            and "![" not in bold_section.group(1)
        ):
            save_section()
            section_lines.clear()

            subsection = bold_section.group(1).strip().rstrip(":")
            section_title = " > ".join(headings + [subsection])
            section_lines.append(line)

        else:
            section_lines.append(line)

    save_section()
    return chunks, {}


def parse_text(path):
    text = path.read_text(encoding="utf-8").strip()
    chunks = [text_chunk("text-0001", text, path.name, "Document")]
    return chunks, {}


def parse_docx(path, output_directory):
    from docx import Document

    chunks = []
    document = Document(path)
    section = "Document"
    section_lines = []
    number = 0

    def save_section():
        nonlocal number
        text = "\n".join(section_lines).strip()
        if text:
            number += 1
            chunks.append(
                text_chunk(
                    f"text-{number:04d}",
                    text,
                    path.name,
                    section,
                )
            )

    for paragraph in document.paragraphs:
        if paragraph.style.name.startswith("Heading"):
            save_section()
            section_lines.clear()
            section = paragraph.text
        else:
            section_lines.append(paragraph.text)
    save_section()

    media_directory = output_directory / "media"
    with zipfile.ZipFile(path) as archive:
        media_files = [
            name for name in archive.namelist() if name.startswith("word/media/")
        ]
        for image_number, name in enumerate(media_files, 1):
            media_directory.mkdir(exist_ok=True)
            image_path = media_directory / Path(name).name
            image_path.write_bytes(archive.read(name))
            chunks.append(
                image_chunk(
                    f"image-{image_number:04d}",
                    image_path,
                    path.name,
                    "Document image",
                    "",
                )
            )

    return chunks, {}


def add_vision_captions(chunks, cache_path=None):
    from PIL import Image

    client = genai.Client()
    text_chunks = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
        if chunk["modality"] == "text"
    }
    caption_chunks = []
    caption_failures = []
    seen_images = set()
    caption_cache = {}
    if cache_path and cache_path.exists():
        with cache_path.open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                caption_cache[record["image_hash"]] = record["caption"]

    images = [
        chunk for chunk in chunks if chunk["modality"] == "image"
    ]
    for image_number, image in enumerate(images, 1):
        location = image["metadata"].get("section_path", "").casefold()
        if (
            "component list" in location
            or location in {"welcome", "contents", "preface", "first use"}
            or location.startswith("welcome >")
        ):
            continue

        image_path = Path(image["image_path"])
        try:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            with Image.open(image_path) as opened_image:
                width, height = opened_image.size
        except Exception as error:
            caption_failures.append({"image": str(image_path), "error": str(error)})
            continue
        if (
            digest in seen_images
            or width < 80
            or height < 80
            or width * height < 20_000
        ):
            continue
        seen_images.add(digest)

        parent_id = image["metadata"].get("parent_chunk_id")
        nearby_text = text_chunks.get(parent_id, {}).get("text", "")[:4000]

        if digest in caption_cache:
            caption = caption_cache[digest]
        else:
            try:
                caption = create_caption(client, image, nearby_text)
            except Exception as error:
                caption_failures.append(
                    {
                        "image": image["image_path"],
                        "error": str(error),
                    }
                )
                continue
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps({
                        "image_hash": digest,
                        "caption": caption,
                    }, ensure_ascii=False) + "\n")

        print(f"Captioned image {image_number}/{len(images)}")

        if caption.upper() == "SKIP":
            continue

        caption_chunks.append(
            {
                "chunk_id": f"caption-{image['chunk_id']}",
                "modality": "text",
                "text": caption,
                "metadata": {
                    **image["metadata"],
                    "content_type": "image_caption",
                    "image_path": image["image_path"],
                },
            }
        )
        parent = text_chunks.get(parent_id)
        if parent is not None:
            parent["metadata"].setdefault("image_chunk_ids", []).append(
                f"caption-{image['chunk_id']}"
            )
            parent["metadata"].setdefault("image_paths", []).append(
                image["image_path"]
            )

    return chunks + caption_chunks, {
        "vision_captions": len(caption_chunks),
        "caption_failures": caption_failures,
    }


def ingest_document(path: Path, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()

    if extension == ".pdf":
        chunks, details = parse_pdf(path, output_directory)
    elif extension in {".md", ".markdown"}:
        chunks, details = parse_markdown(path, output_directory)
    elif extension == ".txt":
        chunks, details = parse_text(path)
    elif extension == ".docx":
        chunks, details = parse_docx(path, output_directory)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    chunks, caption_details = add_vision_captions(
        chunks,
        output_directory / "caption_cache.jsonl",
    )
    details.update(caption_details)

    chunks_path = output_directory / "chunks.jsonl"
    write_chunks(chunks, chunks_path)

    text_chunks = sum(chunk["modality"] == "text" for chunk in chunks)
    image_chunks = sum(chunk["modality"] == "image" for chunk in chunks)
    failed_pages = details.get("failed_pages", [])

    if text_chunks == 0:
        status = "failed"
    elif failed_pages or details["caption_failures"]:
        status = "completed_with_warnings"
    else:
        status = "completed"

    return {
        "document": path.name,
        "status": status,
        "text_chunks": text_chunks,
        "image_chunks": image_chunks,
        "chunks_path": str(chunks_path),
        **details,
    }
