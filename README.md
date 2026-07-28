# Production Multimodal RAG System

This project is an end-to-end RAG system for technical Arduino documentation.
It includes document ingestion, OCR, structured chunking, Gemini embeddings,
Qdrant storage, hybrid retrieval, reranking, grounded answer generation and
RAGAS evaluation.

## Live demo

Temporary public dashboard:

https://separately-fort-simplified-some.trycloudflare.com

The demo runs on the developer's laptop through a Cloudflare Quick Tunnel.
The reviewer does not need to install Python, Docker, Qdrant or Ollama to use
the link.

The link works only while the following local services are running:

- Qdrant on port `6333`
- FastAPI on port `8000`
- Streamlit on port `8501`
- Cloudflare Tunnel

Because this is a temporary public URL, it may change when the tunnel is
restarted.

## Testing the system

The dashboard is divided into three sections.

### Test RAG

Ask a question and inspect the generated answer and its retrieved sources.

Example:

```text
What voltage and interface does the white LED module use?
```

### Evaluation

Generate answers over the golden dataset, calculate the RAGAS metrics and
inspect the result for each question.

The evaluated metrics are:

- Faithfulness
- Answer relevancy
- Context precision
- Context recall

### Qdrant

Inspect hybrid retrieval results or upload a random document. Uploading a
document automatically executes:

```text
upload → parse/OCR → chunk → embed → Qdrant
```

After the upload finishes, the reviewer can immediately ask questions about
the new document. A separate indexing command is not required.

Supported formats:

- PDF
- Markdown
- TXT
- DOCX
- PPTX
- PNG, JPEG, WebP, TIFF and BMP images

For PDFs, native extraction is attempted first. Pages with fewer than 30
extracted characters use Tesseract OCR. If text still cannot be extracted, the
page is reported in `failed_pages`; it is never silently accepted as an empty
page.

## Architecture

### Ingestion

```text
Document
  → format-specific parser
  → native text extraction or OCR
  → section/page chunks with metadata
  → Gemini embeddings
  → Qdrant
```

Markdown documents are chunked by headings and sections. PDFs are processed
page by page. Images and diagrams are represented by retrieval-focused
captions linked to their original image files.

### Retrieval and generation

```text
Question
  → Gemini multi-query rewriting
  → Gemini vector search + BM25 exact-term search
  → reciprocal rank fusion
  → local MiniLM cross-encoder reranking
  → five retrieved contexts
  → grounded Gemini answer
```

Image captions compete normally with text chunks instead of being forcibly
added to every visual query. When a selected caption belongs to a project with
an authoritative Hardware Connection section, that text is attached to the
caption and takes priority over uncertain visual interpretation.

## API

FastAPI documentation is available locally at:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

- `POST /ingest`: upload, parse, chunk, embed and index a document.
- `POST /index`: manually index an existing chunks directory.
- `POST /search`: inspect hybrid retrieval and reranking.
- `POST /ask`: generate a grounded answer with retrieved contexts.
- `POST /evaluate`: generate predictions over a golden dataset.
- `POST /metrics`: calculate or resume RAGAS metrics.
- `GET /runs`: list saved evaluation runs.
- `GET /runs/{run_name}`: return one run's predictions and metrics.

Example `/ask` request:

```json
{
  "query": "What voltage and interface does the white LED module use?"
}
```

## Evaluation design

The project includes a 100-question golden dataset:

```text
arduino_rag_gold_dataset_final.jsonl
```

Each record contains a question, expected answer, expected retrieved
information, source location and whether visual evidence is required.

RAGAS uses local `qwen3:4b-instruct` through Ollama as the primary judge.
Gemini or OpenRouter is used only to retry missing metric values. Completed
scores are saved incrementally so an interrupted evaluation can resume.

Existing prediction and metric files are stored in `runs/` and can be viewed
without recalculating them.

## Local development

Requirements:

- Python 3.10 or newer
- Docker Desktop
- Tesseract OCR
- Gemini API key
- Ollama only when recalculating RAGAS metrics

Install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
cp .env.example .env
```

Add `GEMINI_API_KEY` to `.env`.

Start Qdrant:

```bash
docker compose up -d
```

Start FastAPI:

```bash
python3 -m uvicorn evaluation.api:app --env-file .env --port 8000
```

Start Streamlit in another terminal:

```bash
python3 -m streamlit run dashboard.py
```

Local interfaces:

- Streamlit: `http://127.0.0.1:8501`
- FastAPI: `http://127.0.0.1:8000/docs`
- Qdrant: `http://127.0.0.1:6333/dashboard`

## Reproducibility

The project contains a prebuilt Qdrant snapshot:

```text
qdrant_snapshot/arduino_rag.snapshot
```

It contains 521 indexed text and image-caption points. Restoring it avoids
repeating the original chunk embedding process.

Qdrant is pinned to version `1.18.3` for snapshot compatibility. The snapshot
SHA-256 is:

```text
44dbd76731c5a1f34416e814c0331078fe25eaf4a383cc40c34fee34a72ed134
```

Secrets are stored in `.env`, which is excluded from version control and must
not be included in the submission.
