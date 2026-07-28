import os
import mimetypes
from pathlib import Path

from google import genai
from google.genai import types

from evaluation.contracts import RAGResponse, RetrievedContext
from gemini_generation import generate
from retrieval import search


MODEL = "gemini-3.1-flash-lite"


class GeminiRAGAdapter:
    def answer(self, question: str, record_id: str) -> RAGResponse:
        results = search(question, limit=5, modality="text")

        contexts = [
            RetrievedContext(
                text=result["text"],
                source_id=result["source_document"],
                location=result["source_location"],
                image_path=result.get("image_path", ""),
            )
            for result in results
        ]

        context_text = "\n\n".join(
            (
                f"Context {number}\n"
                f"Source: {result['source_document']}\n"
                f"Location: {result['source_location']}\n"
                f"Content type: {result.get('content_type', 'document_text')}\n"
                f"Evidence:\n{result['text']}"
            )
            for number, result in enumerate(results, 1)
        )

        prompt = f"""Answer the question using only the retrieved evidence.

Instructions:
- Contexts are ordered from most to least relevant. If the first context
  contains all information requested by the question, answer exclusively from
  the first context and ignore every lower-ranked context.
- Some lower-ranked contexts may describe different components. Do not replace
  an answer from a higher-ranked context with values from another source.
- If several contexts answer an ambiguous question for different components,
  use only the earliest directly answering context. Do not list alternative
  answers for other components or source documents.
- Image captions are valid evidence for visible dimensions, labels, pins,
  connections, tables, charts, and diagrams.
- Prefer authoritative document text only when it describes the same component
  and explicitly contradicts an image caption.
- Missing information in an unrelated context is not a conflict.
- Preserve numbers, units, pin names, and technical labels exactly.
- If no context contains the answer, say "I do not know."
- Give a concise answer without discussing the retrieval process.

Question:
{question}

Retrieved evidence:
{context_text}
"""

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        contents = [prompt]
        for context in contexts:
            if context.image_path:
                path = Path(context.image_path)
                contents.append(
                    types.Part.from_bytes(
                        data=path.read_bytes(),
                        mime_type=mimetypes.guess_type(path)[0],
                    )
                )

        response = generate(client, MODEL, contents)

        return RAGResponse(answer=response.text, contexts=contexts)


adapter = GeminiRAGAdapter()
