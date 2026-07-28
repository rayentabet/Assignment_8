# Production Multimodal RAG System

## Final Technical Report

**Project:** Advanced Retrieval-Augmented Generation for Arduino Technical Documentation  
**Evaluation set:** 100-question golden dataset; final reported run uses the first 40 questions  
**Vector database:** Qdrant  
**Embedding model:** Gemini Embedding 2  
**Answer model:** Gemini 3.1 Flash Lite  
**Evaluation framework:** RAGAS  

---

## 1. Executive Summary

This project developed a production-oriented Retrieval-Augmented Generation
(RAG) system for technical Arduino documentation. The goal was not only to
produce answers, but to construct a complete and testable pipeline that can
ingest heterogeneous documents, recover text from scanned pages, preserve
images and diagrams, retrieve evidence through complementary search methods,
generate grounded answers, and measure quality over a fixed golden dataset.

The first dense-vector baseline 
already achieved strong faithfulness and context recall, but its context
precision was only 0.605. Inspection showed that the correct evidence was often
present alongside generic or unrelated sections. Hybrid vector and BM25
retrieval, Reciprocal Rank Fusion (RRF), multi-query rewriting, and local
cross-encoder reranking improved ranking precision, but initially reduced
recall because relevant sibling sections were sometimes removed from the final
top five.

The most important qualitative observation was that several failed questions
depended on wiring diagrams. Text-only retrieval could identify the correct
project but could not always recover the pin mapping visible in an image.
Images were therefore converted into searchable technical captions while
retaining links to the original files. A first caption pilot raised context
precision to 0.802 but reduced faithfulness and recall. Detailed,
retrieval-focused captions produced a better balance: faithfulness 0.944,
answer relevancy 0.875, context precision 0.790, and context recall 0.967 on the
same 15-question subset.

The final architecture was then tested on 40 questions. It achieved:

| Metric | Final mean | Completed judgments |
|---|---:|---:|
| Faithfulness | 0.903 | 40/40 |
| Answer relevancy | 0.880 | 39/40 |
| Context precision | 0.786 | 40/40 |
| Context recall | 0.950 | 40/40 |

These results show that the precision improvement generalized to a larger
question set while recall remained high. The result is not perfect: two
questions still failed because the correct white-LED and traffic-light
evidence was not included in the final context set. These failures are reported
as limitations.

---

## 2. Assignment Requirements and Implemented System

The completed system addresses the required end-to-end workflow.

| Requirement | Implementation |
|---|---|
| Ingest documents through an API | `POST /ingest` accepts PDF, Markdown, TXT, DOCX, PPTX and common image formats |
| Handle scanned or image-only pages | Native PDF extraction is attempted first; pages with fewer than 30 characters use Tesseract OCR |
| Avoid silent empty pages | Unreadable pages are returned explicitly in `failed_pages` |
| Chunk documents | Markdown uses headings/sections; PDFs use page-based text and image children; Office documents preserve available structure |
| Embed and store | Text and image captions use 768-dimensional Gemini embeddings stored in Qdrant with cosine distance |
| Retrieve | Dense search and BM25 are fused with RRF and reranked by a local cross-encoder |
| Generate grounded answers | Gemini answers only from the supplied contexts and returns source information |
| Handle diagrams | Technical images receive searchable captions linked to the original image |
| REST API | FastAPI exposes ingestion, indexing, search, question answering and evaluation endpoints |
| Evaluation | A 100-question golden dataset and Streamlit evaluation dashboard support repeatable experiments |

Uploading a new document through the dashboard now performs the complete write
path automatically:

```text
Upload → Parse/OCR → Chunk → Embed → Qdrant
```

The reviewer can ask a question about the new document immediately without
calling a second indexing endpoint.

---

## 3. Data and Ingestion Design

### 3.1 Source data

The primary evaluated source is a structured Markdown manual for a Keyestudio
37-in-1 Arduino sensor kit. It contains project descriptions, specifications,
hardware-connection instructions, sample-code sections, experiment results,
product images, and wiring diagrams. Additional Arduino PDF material is
included to test page extraction and OCR behavior.

### 3.2 Chunking

Markdown is divided at heading and bold subsection boundaries. A text chunk
stores:

- `chunk_id`
- `source_document`
- `source_location`
- section text

An image chunk stores:

- its original file path
- source document and section
- the identifier of its parent text chunk

PDFs are processed page by page. Every page is rendered as an image so that
layout and diagrams are preserved. Native text is used when available;
otherwise OCR is attempted. This design satisfies the robustness requirement
without silently converting an unreadable scanned page into a successful empty
chunk.

### 3.3 Image representation

Raw images are not embedded as if they were text. A vision-language model
creates a retrieval-focused caption containing the component, purpose,
technical labels, visible connections, and explicit uncertainty. The caption
is embedded as a text child and retains the original `image_path`.

Decorative images, logos, thumbnails, and very small images are skipped.
Captions are cached by SHA-256 hash to avoid repeating vision calls.

---

## 4. Evaluation Method

### 4.1 Golden dataset

The project includes 100 golden records. Each record contains:

- question identifier
- question
- expected answer
- expected retrieved information
- source document and location
- whether visual evidence is required
- related image path when applicable

The initial experiments used the same first 15 questions to support controlled
architecture comparisons. The final generalization run used 40 questions

### 4.2 Metrics

Four RAGAS metrics were used:

| Metric | Interpretation |
|---|---|
| Faithfulness | Whether answer claims are supported by retrieved evidence |
| Answer relevancy | Whether the answer addresses the question directly |
| Context precision | Whether useful evidence is ranked ahead of noise |
| Context recall | Whether the context set contains the evidence required by the reference |

Qwen3 4B Instruct through Ollama was the primary local judge. Gemini or
OpenRouter was used only to retry missing structured judgments. Scores were
saved incrementally so rate limits or evaluator failures did not erase
completed work.

One final answer-relevancy judgment is missing because the evaluator failed to
produce a valid score. It is reported as N/A and is not replaced with zero.

### 4.3 Latency

For the 40-question final prediction run:

| Statistic | Answer latency |
|---|---:|
| Mean | 13.27 s |
| Median | 13.57 s |
| Minimum | 10.44 s |
| Maximum | 16.46 s |

The system prioritizes evaluation quality and free-tier compatibility over
interactive latency. Query rewriting, embedding, hybrid retrieval, reranking,
and grounded generation all contribute to the response time.

---

## 5. Baseline Architecture

The first architecture used structured text chunks, Gemini embeddings, Qdrant
dense retrieval, and a top-five grounded Gemini answer.

```text
Document → structured text chunks → Gemini embeddings → Qdrant

Question → Gemini query embedding
         → top-5 dense text results
         → grounded Gemini answer
```

### 5.1 Baseline results

The completed baseline metric file contains:

| Metric | Mean | Coverage |
|---|---:|---:|
| Faithfulness | 0.904 | 15/15 |
| Answer relevancy | 0.878 | 15/15 |
| Context precision | 0.605 | 15/15 |
| Context recall | 0.967 | 15/15 |

### 5.2 Observation

The baseline usually retrieved enough information, which explains the strong
recall. However, the correct evidence was often mixed with generic
specification or hardware sections from other projects. Context precision was
therefore the main bottleneck.

The white-LED compound question showed two problems. It required both wiring
instructions and experiment behavior, but the final context set did not always
contain both sections. Diagram-dependent questions also exposed a structural
limitation: indexing only text could not reliably recover facts visible only
in wiring images.

### 5.3 Decision

The next architecture needed complementary exact-term retrieval and a stronger
final ranking stage. It also needed a way to retrieve visual evidence.

---

## 6. Hybrid Retrieval Experiment

The retrieval layer was changed while keeping the ingestion data, embeddings,
Qdrant collection, answer model, and final five-context limit fixed.

The added components were:

1. Gemini query rewriting and decomposition into up to three retrieval
   queries.
2. Dense vector retrieval for semantic similarity.
3. BM25 for exact component names, model numbers, pins, voltages and
   protocols.
4. Reciprocal Rank Fusion to combine dense and lexical ranks without directly
   comparing incompatible score scales.
5. A local `cross-encoder/ms-marco-MiniLM-L6-v2` reranker.

```text
Question → multi-query analysis
         ├─→ dense vector candidates
         └─→ BM25 candidates
                ↓
         Reciprocal Rank Fusion
                ↓
         MiniLM cross-encoder reranking
                ↓
         top-5 contexts → grounded answer
```

### 6.1 Result and observation

| Metric | Dense baseline | Hybrid BM25 | Change |
|---|---:|---:|---:|
| Faithfulness | 0.904 | 0.859 | -0.045 |
| Answer relevancy | 0.878 | 0.805 | -0.073 |
| Context precision | 0.605 | 0.702 | +0.097 |
| Context recall | 0.967 | 0.833 | -0.134 |

Hybrid retrieval achieved its primary goal: context precision improved by
0.097. Exact technical terms and cross-encoder rescoring helped rank relevant
sections ahead of generic text.

The improvement introduced a recall trade-off. Query expansion increased the
candidate pool, and the final top-five cutoff sometimes removed another
section required by a compound question. The grounded generator could not
recover a missing fact, so answer quality fell downstream.

### 6.2 Decision

BM25, RRF and reranking were retained because they measurably improved ranking.
The next work focused on restoring recall and representing images rather than
returning to dense-only search.

---

## 7. Multimodal Caption Experiments

### 7.1 Initial image-caption pilot

The failure analysis showed that wiring and pin questions frequently depended
on images. A pilot therefore generated searchable captions for technical
images, embedded those captions, and linked each caption to its original image
and parent text.

| Metric | Image-caption pilot | Coverage |
|---|---:|---:|
| Faithfulness | 0.822 | 15/15 |
| Answer relevancy | 0.826 | 15/15 |
| Context precision | 0.802 | 15/15 |
| Context recall | 0.867 | 15/15 |

Context precision rose substantially, confirming that captions made diagrams
retrievable. However, faithfulness and recall decreased. Inspection showed
that a caption could describe the correct component while omitting a required
pin or misreading dense wiring alignment.

### 7.2 Detailed caption pilot

The caption prompt was then made more technical and conservative:

- preserve visible labels and values exactly
- include pin mappings and connection purpose
- avoid inferring connections from wire color alone
- state uncertainty instead of guessing
- treat nearby document text as authoritative
- skip decorative images

The resulting 15-question score was:

| Metric | Detailed captions | Change from baseline |
|---|---:|---:|
| Faithfulness | 0.944 | +0.040 |
| Answer relevancy | 0.875 | -0.003 |
| Context precision | 0.790 | +0.185 |
| Context recall | 0.967 | +0.000 |

This was the strongest balanced pilot. It preserved baseline recall, increased
faithfulness, and improved context precision by 0.185 with essentially
unchanged answer relevancy.

### 7.3 Caption-routing correction

One intermediate implementation forcibly inserted the single highest-scoring
caption whenever a query contained words such as *diagram*, *wiring*, *pin*,
or *connect*. The larger 40-question inspection showed that this rule could
select a diagram from the wrong board version or even a different project.

The final correction was:

1. Remove global caption forcing.
2. Let captions compete normally in hybrid retrieval and reranking.
3. When a selected caption belongs to a project with a Hardware Connection
   section, attach that authoritative text.
4. In the generation prompt, explicitly prefer authoritative document text
   over a conflicting visual caption.

This retains visual recall without treating a generated caption as a perfect
source of truth.

---

## 8. Architecture Evolution Summary

| Stage | Observation | Architecture modification | Measured outcome |
|---|---|---|---|
| Dense baseline | Correct evidence often accompanied by noise | Establish vector-only top-five baseline | High recall 0.967; low precision 0.605 |
| Hybrid BM25 | Technical terms needed exact matching | Multi-query + dense + BM25 + RRF + reranker | Precision +0.097; recall fell to 0.833 |
| Intermediate tuning (v3) | Needed to rebalance hybrid retrieval | Candidate and retrieval tuning retained hybrid components | F 0.911, AR 0.749, CP 0.696, CR 0.867 |
| Image-caption pilot | Wiring diagrams were not searchable as text | Embed captions linked to images | Precision 0.802; weaker faithfulness/recall |
| Detailed-caption pilot | Generic captions omitted or guessed details | Technical, uncertainty-aware caption prompt | F 0.944, CP 0.790, CR 0.967 |
| Final architecture | Forced captions could select wrong board/project | Normal caption competition + authoritative text grounding | 40-question CP 0.786 and CR 0.950 |

The v3 run is retained as an intermediate experimental artifact. Because
multiple retrieval settings changed together, it is not used to claim the
effect of one isolated component.

---

## 9. Final Architecture

### 9.1 Write path

```text
Uploaded document
    ↓
Format router
    ├─ PDF: native extraction → OCR fallback → rendered page image
    ├─ Markdown: heading/section parser + linked images
    ├─ DOCX/PPTX: structured text + extracted media
    └─ Image: OCR + preserved image
    ↓
Text chunks and image relationships with metadata
    ↓
Technical image captions for useful diagrams
    ↓
Gemini Embedding 2 (768 dimensions)
    ↓
Qdrant cosine-vector collection
```

### 9.2 Query path

```text
User question
    ↓
Gemini query correction/decomposition
    ↓
For each query:
    ├─ Gemini vector search (semantic)
    └─ BM25 search (exact technical terms)
    ↓
Reciprocal Rank Fusion
    ↓
Top candidate pool
    ↓
Local MiniLM cross-encoder reranker
    ↓
Top five contexts
    ↓
Attach authoritative Hardware Connection text to selected captions
    ↓
Attach original image when a selected caption has an image path
    ↓
Gemini grounded answer + source metadata
```

### 9.3 Evaluation path

```text
Golden question
    ↓
Full RAG adapter
    ↓
Saved answer, contexts and latency
    ↓
RAGAS:
    faithfulness
    answer relevancy
    context precision
    context recall
    ↓
Incremental CSV + Streamlit dashboard
```

---

## 10. All Experiment Results

All means below are calculated from the saved metric files. Averages with
incomplete coverage use available values only.

| Run | Questions | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---:|---:|---:|---:|---:|
| Dense baseline | 15 | 0.904 | 0.878 | 0.605 | 0.967 |
| Hybrid BM25 + RRF + reranking | 15 | 0.859 | 0.805 | 0.702 | 0.833 |
| Intermediate v3 | 15 | 0.911 | 0.749 | 0.696 | 0.867 |
| Image-caption pilot | 15 | 0.822 | 0.826 | 0.802 | 0.867 |
| Detailed-caption pilot | 15 | 0.944 | 0.875 | 0.790 | 0.967 |
| Final generalization run | 40 | 0.903 | 0.880* | 0.786 | 0.950 |

\* Answer relevancy has 39/40 completed judgments.

### 10.1 Controlled baseline-to-best-pilot comparison

| Metric | Baseline | Detailed-caption pilot | Absolute change |
|---|---:|---:|---:|
| Faithfulness | 0.904 | 0.944 | +0.040 |
| Answer relevancy | 0.878 | 0.875 | -0.003 |
| Context precision | 0.605 | 0.790 | +0.185 |
| Context recall | 0.967 | 0.967 | +0.000 |

This is the most defensible direct comparison because both runs use the same
15 questions. The 40-question final result should be interpreted as a
generalization test rather than compared as if it used an identical sample.

---

## 11. Final 40-Question Results

| ID | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---:|---:|---:|---:|
| KS-P01-01 | 1.000 | 0.899 | 1.000 | 1.000 |
| KS-P01-02 | 0.833 | 0.708 | 0.000 | 0.000 |
| KS-P02-01 | 1.000 | 0.873 | 0.888 | 1.000 |
| KS-P02-02 | 0.250 | 0.950 | 1.000 | 1.000 |
| KS-P03-01 | 1.000 | 0.796 | 0.478 | 1.000 |
| KS-P03-02 | 1.000 | 0.000 | 0.000 | 0.000 |
| KS-P04-01 | 1.000 | 0.925 | 0.867 | 1.000 |
| KS-P04-02 | 1.000 | 0.934 | 0.750 | 1.000 |
| KS-P05-01 | 0.750 | 0.945 | 1.000 | 1.000 |
| KS-P05-02 | 1.000 | 0.974 | 0.804 | 1.000 |
| KS-P06-01 | 1.000 | 0.948 | 1.000 | 1.000 |
| KS-P06-02 | 1.000 | 0.835 | 1.000 | 1.000 |
| KS-P07-01 | 1.000 | 0.920 | 1.000 | 1.000 |
| KS-P07-02 | 1.000 | 0.910 | 0.804 | 1.000 |
| KS-P08-01 | 1.000 | 0.880 | 0.833 | 1.000 |
| KS-P08-02 | 0.800 | 0.898 | 1.000 | 1.000 |
| KS-P09-01 | 1.000 | 0.983 | 1.000 | 1.000 |
| KS-P09-02 | 1.000 | 0.657 | 1.000 | 1.000 |
| KS-P10-01 | 0.333 | 0.971 | 1.000 | 1.000 |
| KS-P10-02 | 1.000 | 0.884 | 0.333 | 1.000 |
| KS-P11-01 | 0.000 | 0.984 | 1.000 | 1.000 |
| KS-P11-02 | 1.000 | 0.880 | 0.500 | 1.000 |
| KS-P12-01 | 0.667 | 0.967 | 1.000 | 1.000 |
| KS-P12-02 | 1.000 | N/A | 1.000 | 1.000 |
| KS-P13-01 | 1.000 | 0.992 | 1.000 | 1.000 |
| KS-P13-02 | 1.000 | 0.860 | 0.833 | 1.000 |
| KS-P14-01 | 1.000 | 0.940 | 1.000 | 1.000 |
| KS-P14-02 | 1.000 | 0.881 | 0.250 | 1.000 |
| KS-P15-01 | 1.000 | 0.983 | 1.000 | 1.000 |
| KS-P15-02 | 1.000 | 0.871 | 0.804 | 1.000 |
| KS-P16-01 | 1.000 | 1.000 | 0.500 | 1.000 |
| KS-P16-02 | 1.000 | 0.964 | 0.500 | 1.000 |
| KS-P17-01 | 1.000 | 0.892 | 1.000 | 1.000 |
| KS-P17-02 | 1.000 | 0.835 | 1.000 | 1.000 |
| KS-P18-01 | 1.000 | 0.943 | 0.917 | 1.000 |
| KS-P18-02 | 1.000 | 0.903 | 0.000 | 1.000 |
| KS-P19-01 | 1.000 | 0.903 | 1.000 | 1.000 |
| KS-P19-02 | 0.500 | 0.784 | 0.806 | 1.000 |
| KS-P20-01 | 1.000 | 0.920 | 1.000 | 1.000 |
| KS-P20-02 | 1.000 | 0.910 | 0.589 | 1.000 |

---

## 12. Final Results and Discussion

### 12.1 Retrieval quality

Context recall reached 0.950. Thirty-eight of forty questions received a
context-recall score of 1.0. This is strong evidence that the final retriever
usually includes the required facts.

Context precision reached 0.786, substantially above the 0.605 dense baseline
on the controlled pilot. The per-question table shows that many questions now
receive precision scores of 1.0, especially direct specification, definition,
and behavior questions. Hybrid retrieval and reranking are therefore retained
in the final design.

Precision remains lower for some multi-part wiring questions because the
system returns several sections from the correct project or different board
variants. For example, Projects 14, 16, and 20 receive full recall but only
partial precision. This means the answer can be correct while the context list
still contains unnecessary evidence.

### 12.2 Generation quality

Faithfulness is 0.903 and answer relevancy is 0.880. Most answers are grounded
and directly address the question. Low faithfulness values do not always imply
retrieval failure. Some answers paraphrase multiple context details or include
an additional interpretation that the judge does not consider explicitly
supported.

Examples include:

- `KS-P02-02`: full context precision and recall, high answer relevancy, but
  faithfulness 0.250.
- `KS-P11-01`: full retrieval scores and answer relevancy 0.984, but
  faithfulness 0.000.

These combinations should be manually reviewed because they may indicate
either an unsupported answer detail or judge instability. RAGAS is a useful
measurement tool, but it is not treated as unquestionable ground truth.

### 12.3 Remaining failure cases

#### KS-P01-02 — white LED wiring and behavior

The expected evidence contains `S → D7`, power and ground wiring, and a
one-second on/off experiment result. The final contexts did not contain the
complete required evidence, producing context precision 0 and recall 0. The
answer used an incorrect pin and unrelated behavior.

This is a parent/sibling retrieval failure: the board-connection subsection was
found, but the authoritative Hardware Connection and Experiment Result
siblings were not both preserved.

#### KS-P03-02 — traffic-light specifications

The expected answer is a digital interface operating at 3.3–5 V. Retrieval
returned traffic-light content but missed the exact Specifications section.
The grounded model correctly refused to invent the missing voltage, producing
answer relevancy 0, context precision 0, and recall 0 while faithfulness
remained 1.

This demonstrates that high faithfulness can coexist with an unhelpful answer:
the generator behaved correctly given incomplete retrieval.

#### KS-P18-02 — sound sensor

This question received faithfulness, answer relevancy and context recall near
1.0 but context precision 0. The required evidence was present and the answer
was correct, so the zero reflects irrelevant ranking or a questionable
precision judgment rather than a complete RAG failure.

### 12.4 Board-version ambiguity

The manual contains V4.0 and Mega 2560 variants with similar project names and
connection vocabulary. Questions that say only “Arduino pins” can be ambiguous
when the expected answer silently assumes V4.0. BM25 cannot infer a missing
board name; it can only reward exact terms that appear in the question.

A production system should either:

- ask a clarification question,
- use a documented default board,
- or require the board version in the query and golden dataset.

This is preferable to silently selecting a board variant.

### 12.5 Evaluator reliability

The local Qwen judge reliably completed most metrics but occasionally failed
the structured JSON schemas used by context precision and recall. Cloud
fallback judges then encountered rate limits. Incremental result persistence
and retry-only-missing behavior were therefore necessary engineering features.

The final report records coverage beside every mean. The missing
answer-relevancy cell is not converted to zero, because an evaluator failure is
not equivalent to a poor system answer.

---

## 13. Final Decisions

### Retained

- Structured section and page chunking
- Native extraction plus OCR fallback
- Gemini embeddings
- Qdrant
- Multi-query rewriting
- Dense and BM25 hybrid retrieval
- Reciprocal Rank Fusion
- Local cross-encoder reranking
- Top-five grounded generation
- Searchable technical image captions
- Links from captions to original images
- Authoritative-text priority when captions conflict
- Incremental evaluation persistence

### Rejected or corrected

- **Dense-only retrieval:** strong recall but insufficient precision.
- **Raw image vectors as the main retrieval representation:** hard to align
  directly with text questions and unnecessary once searchable captions exist.
- **Forcing one caption into every visually worded query:** introduced
  wrong-project and wrong-board evidence.
- **Treating generated captions as authoritative:** captions can misread dense
  wiring alignment.
- **Replacing missing evaluation values with zero:** confuses evaluator failure
  with system failure.

---

## 14. Post-Evaluation Robustness Fixes

The final 40-question RAGAS run was completed before the submission and
runtime-ingestion fixes below were added. These changes improve robustness,
diagnostics and demonstration usability, but they were **not evaluated in a
new controlled run**. Therefore, the scores reported in this document must not
be interpreted as evidence that these fixes improved the core RAG
architecture.

| Fix | Motivation | Validation performed | Effect on reported metrics |
|---|---|---|---|
| Runtime vision captions for uploaded images and PDF pages | OCR extracts characters but loses visual relationships such as arrows, wiring and dimensions | A mounting-hole diagram produced a caption containing the correct 10.2 mm and 11.4 mm center-to-center distances | Not measured; final RAGAS scores are unchanged |
| OCR and vision caption stored as separate text chunks | Preserve exact recognized text while adding a searchable visual interpretation | Confirmed that both chunks were created and the caption retained the original image path | Not measured |
| Caption prompt extended to preserve dimensions and spatial relationships | Generic captions could list values without explaining which holes, arrows or locations they described | The test caption correctly distinguished upper/lower spacing and hole diameters | Single-document functional test only |
| Caption failure reporting and OCR fallback | A vision API failure must not silently discard an otherwise usable document | `/ingest` now reports `vision_captions`, `caption_failures` and `completed_with_warnings` while retaining native/OCR text | Reliability improvement, not a metric result |
| Minimal answer-grounding clarification | Once captions became runtime evidence, the generator needed to recognize them as valid for visible labels, pins, connections and dimensions | One end-to-end smoke test answered the uploaded dimension question from its retrieved caption | Not included in the 40-question evaluation |
| README, snapshot and startup corrections | Make the submission reproducible without repeating corpus embedding | Commands, ports, snapshot contents, ignored secrets and ingestion steps were audited | No architectural or metric effect |

An attempted query-specific ranking prompt was also tested and then reverted.
Although it fixed one ambiguous example, it risked overfitting generation to a
single retrieval order. The retained grounding change is deliberately narrow:
it only states that some contexts may be irrelevant and that image captions
are valid evidence for visible information.

This separation is important for scientific reporting. The evaluated final
architecture remains the system described in Sections 5–13. The additions in
this section should be treated as post-evaluation engineering fixes that
require a future regression run before any improvement claim is made.

---

## 15. Limitations and Future Work

1. **Metadata-aware board filtering.** Extract board names such as V4.0 and
   Mega 2560 into explicit metadata and filter or boost matching candidates.
2. **Sibling expansion.** When a Hardware Connection child is selected,
   retrieve its related Specifications and Experiment Result siblings for
   compound questions.
3. **Selective clarification.** Ask the user which board is intended when
   multiple incompatible variants are equally plausible.
4. **Caption verification.** Use a stronger vision model selectively on
   difficult wiring diagrams, then validate pin mappings against nearby
   authoritative text.
5. **PDF-specific evaluation.** The main 40-question set is concentrated on
   the Markdown manual. A dedicated PDF subset is required before drawing
   conclusions about whole-page PDF chunking.
6. **Latency optimization.** Cache query rewrites and embeddings, avoid
   rewriting simple single-intent questions, and load the reranker once at
   service startup.
7. **Larger evaluation.** Complete the full 100-question run when judge and API
   quotas allow.
8. **Deployment security.** The Cloudflare Quick Tunnel is suitable for a
   supervised demonstration, but a persistent deployment should add
   authentication, upload limits, logging and secret management.

---

## 16. Deployment and Demonstration

The live Streamlit dashboard exposes:

- **Test RAG:** ask a custom question and inspect sources.
- **Evaluation:** generate predictions, calculate RAGAS metrics, and inspect
  each question.
- **Qdrant:** inspect retrieval and upload a random document.

The temporary Cloudflare URL allows the professor to test the application
without installing the project. The laptop continues to run Streamlit,
FastAPI, Qdrant and the tunnel.

For reproducibility, the repository also contains:

- source code and dependency manifest
- the 100-question golden dataset
- experiment predictions and metrics
- extracted chunks and image-caption cache
- a Qdrant 1.18.3 snapshot containing 521 indexed points

The snapshot avoids repeating the original embedding process.

---

## 17. Conclusion

The final system was shaped by measured failures rather than by adding
techniques for their own sake.

The baseline established that dense retrieval could achieve high recall, but
its low context precision showed that semantic similarity alone was
insufficient for repetitive technical documentation. Hybrid BM25 retrieval,
RRF and reranking improved ranking but exposed a recall trade-off. Inspection
then showed that several remaining failures depended on diagrams, motivating
searchable image captions. The first caption experiment improved precision but
introduced incomplete or uncertain evidence. Detailed captions and
authoritative-text grounding recovered faithfulness and recall while
maintaining a large precision improvement.

On the final 40-question test, the system achieved 0.903 faithfulness, 0.880
answer relevancy, 0.786 context precision and 0.950 context recall. The final
precision remains well above the dense baseline's main weakness, and recall
remains close to the original high level. Two retrieval failures and one
missing evaluator score are reported transparently.

The principal lesson is that multimodal retrieval is not solved merely by
adding images. Visual evidence must be converted into searchable,
uncertainty-aware representations, linked to document structure, and checked
against authoritative text. Likewise, advanced retrieval components are
valuable only when their effect is measured on a fixed dataset. This
observation-driven process produced a stronger and more defensible final RAG
architecture.
