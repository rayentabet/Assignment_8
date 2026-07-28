import json
import os
import time
from pathlib import Path


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            result = json.loads(line)
            rows.append(
                {
                    "question": result["query"],
                    "answer": result["answer"],
                    "contexts": [
                        context["text"] for context in result["contexts"]
                    ],
                    "ground_truth": result["expected_answer"],
                }
            )
    return rows


def calculate_metrics(predictions_path: Path, output_path: Path):
    import pandas as pd
    from datasets import Dataset
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
    from openai import RateLimitError
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    rows = load_predictions(predictions_path)
    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model="qwen3:4b-instruct",
            temperature=0,
            num_ctx=8192,
            format="json",
            client_kwargs={"timeout": 600},
            async_client_kwargs={"timeout": 600},
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=os.environ["GEMINI_API_KEY"],
        )
    )
    fallback_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            timeout=600,
        )
    )
    structured_fallback_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=0,
            timeout=600,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_objects = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_recall": context_recall,
        "context_precision": context_precision,
    }

    existing_results = output_path.exists()
    if existing_results:
        results = pd.read_csv(output_path)
    else:
        results = pd.DataFrame(
            {
                "user_input": [row["question"] for row in rows],
                "retrieved_contexts": [row["contexts"] for row in rows],
                "response": [row["answer"] for row in rows],
                "reference": [row["ground_truth"] for row in rows],
            }
        )

    for metric_name, metric in metric_objects.items():
        if metric_name not in results:
            results[metric_name] = float("nan")

        missing_indexes = results.index[results[metric_name].isna()].tolist()
        if not missing_indexes:
            continue

        if not existing_results:
            missing_rows = [rows[index] for index in missing_indexes]
            scores = evaluate(
                Dataset.from_list(missing_rows),
                metrics=[metric],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=RunConfig(
                    timeout=600,
                    max_retries=2,
                    max_workers=1,
                ),
            ).to_pandas()

            for index, score in zip(missing_indexes, scores[metric_name]):
                results.loc[index, metric_name] = score

            results.to_csv(output_path, index=False)

        fallback_indexes = results.index[results[metric_name].isna()].tolist()
        for index in fallback_indexes:
            for attempt in range(5):
                try:
                    if metric_name in {
                        "context_precision",
                        "context_recall",
                    }:
                        time.sleep(5)

                    score = evaluate(
                        Dataset.from_list([rows[index]]),
                        metrics=[metric],
                        llm=(
                            structured_fallback_llm
                            if metric_name
                            in {"context_precision", "context_recall"}
                            else fallback_llm
                        ),
                        embeddings=evaluator_embeddings,
                        run_config=RunConfig(
                            timeout=600,
                            max_retries=1,
                            max_workers=1,
                        ),
                        raise_exceptions=True,
                    ).to_pandas()
                except RateLimitError:
                    break
                except Exception:
                    if attempt < 4:
                        time.sleep(60)
                    continue

                value = score.loc[0, metric_name]

                if pd.notna(value):
                    results.loc[index, metric_name] = value
                    results.to_csv(output_path, index=False)
                    break

                if attempt < 4:
                    time.sleep(60)

    return results
