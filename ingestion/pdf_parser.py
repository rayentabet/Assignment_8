import re
import statistics


TARGET_TOKENS = 700
# Leaves room to repeat the project path on every final chunk.
MAX_TOKENS = 950
OVERLAP_TOKENS = 100

NUMBERED_HEADING = re.compile(
    r"^(?:chapter\s+\d+|(?:\d+\.)*\d+)[\s.:)\-]+\S+", re.IGNORECASE
)


def token_count(text):
    """Estimate tokens without loading a tokenizer."""
    pieces = re.findall(r"\w+|[^\w\s]", text)
    return max(1, round(len(pieces) * 1.15))


def normalize_heading(text):
    """Normalize visible headings so they can be matched to PDF bookmarks."""
    return " ".join(text.casefold().split()).strip(" .:-")


def table_of_contents_headings(document):
    """Map each PDF page to its bookmark titles and hierarchy levels."""
    headings = {}
    for level, title, page_number in document.get_toc():
        if page_number < 1:
            continue
        headings.setdefault(page_number, {})[normalize_heading(title)] = level
    return headings


def is_repeated_margin_text(block, page_height):
    """Remove common running headers, footers and isolated page numbers."""
    top, bottom = block["bbox"][1], block["bbox"][3]
    in_margin = top < page_height * 0.09 or bottom > page_height * 0.93
    if not in_margin:
        return False

    text = " ".join(block["text"].split())
    repeated_label = re.search(
        r"www\.|need help\?|contact\s+\S+@|support@", text, re.IGNORECASE
    )
    return bool(repeated_label or re.fullmatch(r"\d+", text))


def extract_text_blocks(page):
    """Return text blocks with the layout information needed for headings."""
    text_blocks = []

    for block in page.get_text("dict", sort=True).get("blocks", []):
        if block.get("type") != 0:
            continue

        lines = []
        spans = []
        for line in block.get("lines", []):
            line_spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in line_spans)
            if line_text.strip():
                lines.append(line_text.strip())
                spans.extend(line_spans)

        text = "\n".join(lines).strip()
        if not text:
            continue

        font_sizes = [
            span.get("size", 0)
            for span in spans
            if span.get("text", "").strip()
        ]
        text_blocks.append(
            {
                "type": "text",
                "text": text,
                "bbox": block.get("bbox", (0, 0, 0, 0)),
                "font_size": max(font_sizes, default=0),
                "bold": any("bold" in span.get("font", "").lower() for span in spans),
            }
        )

    return text_blocks


def find_body_font_size(blocks):
    """Find the typical body font while giving long blocks more weight."""
    font_sizes = []
    for block in blocks:
        weight = max(1, min(len(block["text"]) // 20, 20))
        font_sizes.extend([round(block["font_size"], 1)] * weight)
    return statistics.median(font_sizes) if font_sizes else 11.0


def heading_level(block, body_font_size):
    """Return a heading level, or None when the block is normal body text."""
    text = " ".join(block["text"].split())
    if not text or len(text) > 160 or text.endswith((".", ";", ",")):
        return None

    numbered = NUMBERED_HEADING.match(text)
    larger_than_body = block["font_size"] >= body_font_size * 1.18
    emphasized_number = numbered and (
        block["bold"] or block["font_size"] >= body_font_size * 1.05
    )
    if not larger_than_body and not emphasized_number:
        return None

    if re.match(r"^chapter\s+\d+", text, re.IGNORECASE):
        return 1

    number = re.match(r"^(\d+(?:\.\d+)*)", text)
    if number:
        return min(number.group(1).count(".") + 1, 6)

    font_ratio = block["font_size"] / max(body_font_size, 1)
    if font_ratio >= 1.6:
        return 1
    if font_ratio >= 1.35:
        return 2
    return 3


def split_oversized_paragraph(paragraph):
    """Split one very large PDF block into pieces below the hard limit."""
    if token_count(paragraph["text"]) <= MAX_TOKENS:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+|\n+", paragraph["text"])
    pieces = []
    current_words = []

    for sentence in sentences:
        for word in sentence.split():
            candidate = " ".join(current_words + [word])
            if current_words and token_count(candidate) > MAX_TOKENS:
                pieces.append({**paragraph, "text": " ".join(current_words)})
                current_words = []
            current_words.append(word)

    if current_words:
        pieces.append({**paragraph, "text": " ".join(current_words)})
    return pieces


def chunk_section(paragraphs):
    """Create bounded chunks without crossing the section boundary."""
    chunks = []
    current = []

    def size(items):
        return token_count("\n\n".join(item["text"] for item in items))

    def overlap(items):
        if items and token_count(items[-1]["text"]) <= OVERLAP_TOKENS:
            return items[-1:]
        return []

    for paragraph in paragraphs:
        for piece in split_oversized_paragraph(paragraph):
            if current and size(current + [piece]) > MAX_TOKENS:
                chunks.append(current)
                current = overlap(current)
                if size(current + [piece]) > MAX_TOKENS:
                    current = []

            current.append(piece)
            if size(current) >= TARGET_TOKENS:
                chunks.append(current)
                current = overlap(current)

    overlap_only = chunks and len(current) == 1 and current[0] is chunks[-1][-1]
    if current and not overlap_only:
        chunks.append(current)
    return chunks


def project_path(section_path):
    """Return the chapter/project path used to group small sibling sections."""
    parts = section_path.split(" > ")
    for index, part in enumerate(parts):
        if part.casefold().startswith("project "):
            return " > ".join(parts[: index + 1])
    return None


def pack_project_sections(sections):
    """Combine consecutive subsections without crossing a project boundary."""
    packed = []

    for section in sections:
        group_path = project_path(section["path"])
        can_join_previous = (
            group_path
            and packed
            and packed[-1].get("group_path") == group_path
        )

        if not can_join_previous:
            packed.append({
                "id": section["id"],
                "path": group_path or section["path"],
                "group_path": group_path,
                "section_paths": [],
                "paragraphs": [],
                "images": [],
            })

        group = packed[-1]
        content_pages = [item["page"] for item in section["paragraphs"] + section["images"]]
        heading_page = min(content_pages) if content_pages else section["page_end"]
        group["section_paths"].append(section["path"])
        group["paragraphs"].append({
            "text": section["path"],
            "page": heading_page,
        })
        group["paragraphs"].extend(section["paragraphs"])
        group["images"].extend(section["images"])

    return packed


def parse_pdf(path, output_directory, text_chunk, image_chunk):
    """Extract a PDF into cross-page section chunks and related images."""
    import fitz
    import pytesseract
    from PIL import Image

    pages_directory = output_directory / "pages"
    figures_directory = output_directory / "figures"
    pages_directory.mkdir(exist_ok=True)
    figures_directory.mkdir(exist_ok=True)

    document = fitz.open(path)
    toc_headings = table_of_contents_headings(document)
    pages = []
    native_blocks = []
    failed_pages = []
    native_page_count = 0
    ocr_page_count = 0

    for page_number, page in enumerate(document, 1):
        blocks = extract_text_blocks(page)
        blocks = [
            block
            for block in blocks
            if not is_repeated_margin_text(block, page.rect.height)
        ]
        native_text = "\n".join(block["text"] for block in blocks)
        rendered_page = None

        if len(native_text.strip()) < 30:
            rendered_page = pages_directory / f"page-{page_number}.png"
            page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(rendered_page)
            ocr_text = pytesseract.image_to_string(Image.open(rendered_page)).strip()
            blocks = [{
                "type": "text",
                "text": ocr_text,
                "bbox": (0, 0, page.rect.width, page.rect.height),
                "font_size": 11.0,
                "bold": False,
            }] if ocr_text else []
            ocr_page_count += 1
        else:
            native_blocks.extend(blocks)
            native_page_count += 1

        if not blocks:
            failed_pages.append(page_number)

        images = []
        for image_number, image_info in enumerate(page.get_images(full=True), 1):
            try:
                extracted = document.extract_image(image_info[0])
            except Exception:
                continue

            extension = extracted.get("ext", "png")
            image_path = figures_directory / (
                f"page-{page_number:04d}-image-{image_number:02d}.{extension}"
            )
            image_path.write_bytes(extracted["image"])
            rectangles = page.get_image_rects(image_info[0])
            bbox = tuple(rectangles[0]) if rectangles else (0, 0, 0, page.rect.height)
            images.append({"type": "image", "path": image_path, "bbox": bbox})

        if rendered_page and not images:
            images.append({
                "type": "image",
                "path": rendered_page,
                "bbox": (0, 0, page.rect.width, page.rect.height),
            })

        elements = blocks + images
        elements.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        pages.append({"number": page_number, "elements": elements})

    body_font_size = find_body_font_size(native_blocks)
    sections = []
    heading_stack = []
    current_section = None

    def new_section(section_path, page_number):
        section = {
            "id": f"section-{len(sections) + 1:04d}",
            "path": section_path,
            "paragraphs": [],
            "images": [],
            "page_end": page_number,
        }
        sections.append(section)
        return section

    for page in pages:
        for element in page["elements"]:
            if element["type"] == "text":
                if toc_headings:
                    level = toc_headings.get(page["number"], {}).get(
                        normalize_heading(element["text"])
                    )
                else:
                    level = heading_level(element, body_font_size)
                if level:
                    heading_stack = heading_stack[: level - 1]
                    heading_stack.append(" ".join(element["text"].split()))
                    current_section = new_section(" > ".join(heading_stack), page["number"])
                    continue

            if current_section is None:
                current_section = new_section("Document introduction", page["number"])

            if element["type"] == "text":
                current_section["paragraphs"].append({
                    "text": element["text"],
                    "page": page["number"],
                })
            else:
                current_section["images"].append({
                    "path": element["path"],
                    "page": page["number"],
                    "section_path": current_section["path"],
                })
            current_section["page_end"] = page["number"]

    packed_sections = pack_project_sections(sections)
    chunks = []
    for section in packed_sections:
        text_chunks = []
        for part_number, paragraphs in enumerate(chunk_section(section["paragraphs"]), 1):
            pages_in_chunk = [paragraph["page"] for paragraph in paragraphs]
            content = "\n\n".join(paragraph["text"] for paragraph in paragraphs)
            if not content.startswith(section["path"]):
                content = f"{section['path']}\n\n{content}"
            chunk = text_chunk(
                f"{section['id']}-part-{part_number:02d}",
                content,
                path.name,
                section["path"],
            )
            chunk["metadata"].update({
                "section_id": section["id"],
                "section_path": section["path"],
                "section_paths": section["section_paths"],
                "page_start": min(pages_in_chunk),
                "page_end": max(pages_in_chunk),
                "part_number": part_number,
                "content_type": "section_text",
                "image_chunk_ids": [],
                "image_paths": [],
            })
            chunks.append(chunk)
            text_chunks.append(chunk)

        for image_number, image in enumerate(section["images"], 1):
            matching_chunks = [
                chunk for chunk in text_chunks
                if chunk["metadata"]["page_start"] <= image["page"] <= chunk["metadata"]["page_end"]
            ]
            parent = (
                matching_chunks[0]
                if matching_chunks
                else (text_chunks[0] if text_chunks else None)
            )
            chunk = image_chunk(
                f"{section['id']}-image-{image_number:02d}",
                image["path"],
                path.name,
                image["section_path"],
                parent["chunk_id"] if parent else "",
                image["page"],
            )
            chunk["metadata"].update({
                "section_id": section["id"],
                "section_path": image["section_path"],
                "project_path": section["path"],
                "content_type": "section_image",
            })
            chunks.append(chunk)

    return chunks, {
        "pages": len(document),
        "sections": len(sections),
        "packed_sections": len(packed_sections),
        "native_pages": native_page_count,
        "ocr_pages": ocr_page_count,
        "failed_pages": failed_pages,
    }
