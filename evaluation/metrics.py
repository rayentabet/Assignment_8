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
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
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
        ChatOpenAI(
            model=os.getenv(
                "RAGAS_EVALUATOR_MODEL",
                "nvidia/nemotron-3-ultra-550b-a55b",
            ),
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=2048,
            timeout=600,
            max_retries=2,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=os.environ["GEMINI_API_KEY"],
        )
    )
    local_fallback_llm = LangchainLLMWrapper(
        ChatOllama(
            model="qwen3:4b-instruct",
            temperature=0,
            num_ctx=8192,
            format="json",
            client_kwargs={"timeout": 600},
            async_client_kwargs={"timeout": 600},
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_objects = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_recall": context_recall,
        "context_precision": context_precision,
    }
    evaluator_rpm = int(os.getenv("RAGAS_REQUESTS_PER_MINUTE", "10"))
    if evaluator_rpm < 1:
        raise ValueError("RAGAS_REQUESTS_PER_MINUTE must be at least 1")
    evaluator_interval = 60 / evaluator_rpm

    if output_path.exists():
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

        for index in results.index[results[metric_name].isna()].tolist():
            value = float("nan")
            time.sleep(evaluator_interval)
            try:
                score = evaluate(
                    Dataset.from_list([rows[index]]),
                    metrics=[metric],
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    run_config=RunConfig(
                        timeout=600,
                        max_retries=1,
                        max_workers=1,
                    ),
                    raise_exceptions=True,
                ).to_pandas()
                value = score.loc[0, metric_name]
            except Exception as error:
                print(
                    f"Nemotron RAGAS failed for {metric_name} "
                    f"row {index}: {error}"
                )

            if pd.isna(value):
                print(f"Using local Ollama fallback for {metric_name} row {index}")
                try:
                    score = evaluate(
                        Dataset.from_list([rows[index]]),
                        metrics=[metric],
                        llm=local_fallback_llm,
                        embeddings=evaluator_embeddings,
                        run_config=RunConfig(
                            timeout=600,
                            max_retries=1,
                            max_workers=1,
                        ),
                        raise_exceptions=True,
                    ).to_pandas()
                    value = score.loc[0, metric_name]
                except Exception as error:
                    print(
                        f"Local RAGAS fallback failed for {metric_name} "
                        f"row {index}: {error}"
                    )

            if pd.notna(value):
                results.loc[index, metric_name] = value

            results.to_csv(output_path, index=False)
            print(f"Saved {metric_name} row {index}: {value}")

    return results
