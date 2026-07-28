import os


os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"


MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
reranker = None


def candidate_text(candidate):
    location = candidate.get("source_location", "")
    location_parts = [
        part.strip() for part in location.split(">") if part.strip()
    ]
    project = next(
        (
            part
            for part in location_parts
            if part.lower().startswith("project")
        ),
        "",
    )
    section = location_parts[-1] if location_parts else location

    return (
        f"Document: {candidate.get('source_document', '')}\n"
        f"Project: {project}\n"
        f"Section: {section}\n"
        f"Location: {location}\n"
        f"Content:\n{candidate['text']}"
    )


def rerank(query, candidates, limit):
    global reranker
    from sentence_transformers import CrossEncoder

    if reranker is None:
        reranker = CrossEncoder(MODEL)

    pairs = [
        (query, candidate_text(candidate))
        for candidate in candidates
    ]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    return sorted(
        candidates,
        key=lambda candidate: candidate["rerank_score"],
        reverse=True,
    )[:limit]
