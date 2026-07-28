# Production Multimodal RAG System

This project is an end-to-end RAG system for technical Arduino documentation.
It includes document ingestion, OCR, structured chunking, Gemini embeddings,
Qdrant storage, hybrid retrieval, reranking, grounded answer generation and
RAGAS evaluation.

## Demo access

When the temporary public tunnel is active, the dashboard is available at:

https://separately-fort-simplified-some.trycloudflare.com

The URL is a Cloudflare Quick Tunnel to the application running on the
developer's laptop. It is not a permanent deployment, so it works only while
the laptop and the services below are running. If the URL is unavailable, use
the local setup instructions in this README.

Required services:

- Qdrant on port `6333`
- FastAPI on port `8000`
- Streamlit on port `8501`
- Cloudflare Tunnel

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

For PDFs, native extraction is attempted first. A page with fewer than 30
extracted characters is rendered and passed to Tesseract OCR. If neither
method extracts text, its page number appears in `failed_pages`. A document
with no extractable text returns `status: "failed"` and indexes zero chunks
instead of silently succeeding.

The uploaded file and generated chunks are saved under ignored runtime
directories (`uploads/` and `ingested/`). They are not added to Git.

### Quick ingestion test

Before uploading, confirm that all three local interfaces open:

- FastAPI health/docs: `http://127.0.0.1:8000/docs`
- Qdrant: `http://127.0.0.1:6333/dashboard`
- Streamlit: `http://127.0.0.1:8501`

In Streamlit, open **Qdrant → Ingest document**, select a file and click
**Ingest document**. A successful response contains:

```json
{
  "status": "completed",
  "text_chunks": 1,
  "failed_pages": [],
  "indexing": {
    "indexed_chunks": 1
  }
}
```

Counts vary by document. After ingestion, use **Test RAG** to ask a question
whose answer appears in the uploaded document.

The same test can be performed directly through FastAPI:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@/absolute/path/to/document.pdf"
```

If the result contains `completed_with_warnings`, inspect `failed_pages`. If
it contains `failed`, verify that Tesseract is installed and that the scanned
pages are readable.

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
page by page.

For the supplied Arduino corpus, images and wiring diagrams were processed
offline with a vision-language model. The resulting detailed captions are
embedded as text and retain a path to the original image. This is the
multimodal caption pipeline evaluated in the report.

Runtime uploads through `/ingest` use native extraction and OCR, then index
the resulting text chunks. Their extracted image/page files are retained for
inspection, but new vision captions are not generated automatically during
the request. This keeps random-document ingestion fast and avoids one
vision-model API call per page.

### Retrieval and generation

```text
Question
  → Gemini multi-query rewriting
  → Gemini vector search + BM25 exact-term search
  → reciprocal rank fusion
  → local MiniLM cross-encoder reranking
  → up to five retrieved contexts
  → grounded Gemini answer
```

Precomputed image captions compete normally with text chunks instead of being
forcibly added to every visual query. When a selected caption belongs to a
project with an authoritative Hardware Connection section, that text is
attached to the caption and takes priority over uncertain visual
interpretation.

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

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
cp .env.example .env
```

Add `GEMINI_API_KEY` to `.env`.

### 2. Start Qdrant

```bash
docker compose up -d
```

Wait until `http://127.0.0.1:6333/dashboard` opens.

### 3. Start FastAPI

```bash
python3 -m uvicorn evaluation.api:app --env-file .env --port 8000
```

Wait for `Application startup complete`, then leave this terminal running.

### 4. Start Streamlit

In another terminal, activate the same virtual environment and run:

```bash
source .venv/bin/activate
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

To use the prepared index, start Qdrant and restore
`qdrant_snapshot/arduino_rag.snapshot` into a collection named
`arduino_rag`. This can be done from the Qdrant dashboard's snapshot controls
or through Qdrant's snapshot-upload API. The application reads that collection
name from `QDRANT_COLLECTION` in `.env`.

If the snapshot is not restored, the collection is created automatically the
first time `/ingest` or `/index` indexes chunks. Indexing the complete supplied
corpus again consumes Gemini embedding quota, so snapshot restoration is the
recommended reviewer path.

Secrets are stored in `.env`, which is excluded from version control and must
not be included in the submission.

## Repository contents

- `ingestion/`: parsing, OCR and chunk creation
- `embedding/`: Gemini embedding and Qdrant indexing
- `retrieval/`: multi-query, BM25/vector fusion and reranking
- `evaluation/`: FastAPI endpoints and RAGAS evaluation
- `dashboard.py`: Streamlit test, evaluation and Qdrant interface
- `arduino_rag_gold_dataset_final.jsonl`: 100-question golden dataset
- `runs/`: saved experiment predictions and metrics
- `qdrant_snapshot/`: prepared vector-store snapshot and checksum
- `report/`: final report in Markdown and DOCX formats
