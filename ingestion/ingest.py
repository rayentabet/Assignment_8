import json
import re
import shutil
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
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
    import fitz
    import pytesseract
    from PIL import Image

    chunks = []
    failed_pages = []
    native_pages = 0
    ocr_pages = 0
    pages_directory = output_directory / "pages"
    pages_directory.mkdir(exist_ok=True)

    document = fitz.open(path)

    for page_index, page in enumerate(document):
        page_number = page_index + 1
        text = page.get_text().strip()

        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        page_image_path = pages_directory / f"page-{page_number}.png"
        pixmap.save(page_image_path)

        if len(text) < 30:
            text = pytesseract.image_to_string(Image.open(page_image_path)).strip()
            ocr_pages += 1
        else:
            native_pages += 1

        parent_id = f"text-page-{page_number:04d}"

        if text:
            chunks.append(
                text_chunk(
                    parent_id,
                    text,
                    path.name,
                    f"Page {page_number}",
                    page_number,
                )
            )
        else:
            failed_pages.append(page_number)
            parent_id = ""

        chunks.append(
            image_chunk(
                f"image-page-{page_number:04d}",
                page_image_path,
                path.name,
                f"Page {page_number}",
                parent_id,
                page_number,
            )
        )

    return chunks, {
        "pages": len(document),
        "native_pages": native_pages,
        "ocr_pages": ocr_pages,
        "failed_pages": failed_pages,
    }


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


def parse_pptx(path, output_directory):
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    chunks = []
    presentation = Presentation(path)
    media_directory = output_directory / "media"

    for slide_number, slide in enumerate(presentation.slides, 1):
        lines = [shape.text for shape in slide.shapes if hasattr(shape, "text_frame")]
        text = "\n".join(lines).strip()
        parent_id = f"text-slide-{slide_number:04d}"

        if text:
            chunks.append(
                text_chunk(
                    parent_id,
                    text,
                    path.name,
                    f"Slide {slide_number}",
                    slide_number,
                )
            )

        image_number = 0
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                image_number += 1
                media_directory.mkdir(exist_ok=True)
                extension = shape.image.ext
                image_path = (
                    media_directory / f"slide-{slide_number}-image-{image_number}.{extension}"
                )
                image_path.write_bytes(shape.image.blob)
                chunks.append(
                    image_chunk(
                        f"image-slide-{slide_number:04d}-{image_number}",
                        image_path,
                        path.name,
                        f"Slide {slide_number}",
                        parent_id if text else "",
                        slide_number,
                    )
                )

    return chunks, {"pages": len(presentation.slides)}


def parse_image(path, output_directory):
    import pytesseract
    from PIL import Image

    image_path = output_directory / path.name
    shutil.copy2(path, image_path)
    text = pytesseract.image_to_string(Image.open(path)).strip()
    chunks = []
    failed_pages = []
    parent_id = ""

    if text:
        parent_id = "text-0001"
        chunks.append(text_chunk(parent_id, text, path.name, "Image", 1))
    else:
        failed_pages.append(1)

    chunks.append(
        image_chunk("image-0001", image_path, path.name, "Image", parent_id, 1)
    )
    return chunks, {
        "pages": 1,
        "native_pages": 0,
        "ocr_pages": 1,
        "failed_pages": failed_pages,
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
    elif extension == ".pptx":
        chunks, details = parse_pptx(path, output_directory)
    elif extension in IMAGE_EXTENSIONS:
        chunks, details = parse_image(path, output_directory)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    chunks_path = output_directory / "chunks.jsonl"
    write_chunks(chunks, chunks_path)

    text_chunks = sum(chunk["modality"] == "text" for chunk in chunks)
    image_chunks = sum(chunk["modality"] == "image" for chunk in chunks)
    failed_pages = details.get("failed_pages", [])

    if text_chunks == 0:
        status = "failed"
    elif failed_pages:
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
