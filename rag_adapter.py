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
            f"Context {number}:\n{context.text}"
            for number, context in enumerate(contexts, 1)
        )

        prompt = f"""Answer the question using only the provided contexts.
If the answer is not in the contexts, say that you do not know.
When authoritative document text conflicts with an image caption, follow the
authoritative document text. Use image captions only for extra visual details.
Keep the answer concise.

Question:
{question}

Contexts:
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
