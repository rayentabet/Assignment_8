import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="RAG Dashboard", layout="wide")
st.title("RAG Dashboard")

api_url = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")
test_tab, evaluation_tab, qdrant_tab = st.tabs(
    ["Test RAG", "Evaluation", "Qdrant"]
)


with test_tab:
    st.header("Ask the RAG")
    question = st.text_input("Ask a question about the indexed documents")

    if st.button("Generate answer", disabled=not question):
        with st.spinner("Retrieving evidence and generating an answer..."):
            response = requests.post(
                f"{api_url}/ask",
                json={"query": question},
            )
        if response.ok:
            st.session_state["rag_answer"] = response.json()
        else:
            st.error(response.text)

    if "rag_answer" in st.session_state:
        rag_answer = st.session_state["rag_answer"]
        st.subheader("Answer")
        st.write(rag_answer["answer"])
        st.subheader("Sources")

        for number, context in enumerate(rag_answer["contexts"], 1):
            title = f"{number}. {context['source_id']} — {context['location']}"
            with st.expander(title):
                st.write(context["text"])


with evaluation_tab:
    st.header("Run evaluation")

    setup_columns = st.columns(2)
    with setup_columns[0]:
        dataset_path = st.text_input(
            "Golden dataset",
            "arduino_rag_gold_dataset_final.jsonl",
        )
        run_name = st.text_input("Run name", "hybrid_bm25")

    with setup_columns[1]:
        adapter = st.text_input("RAG adapter", "rag_adapter:adapter")
        limit = st.number_input("Question limit", min_value=1, value=5)

    if st.button("Generate predictions"):
        with st.spinner(f"Generating answers for {limit} questions..."):
            response = requests.post(
                f"{api_url}/evaluate",
                json={
                    "dataset_path": dataset_path,
                    "adapter": adapter,
                    "output_dir": f"runs/{run_name}",
                    "limit": limit,
                },
            )
        if response.ok:
            st.success(response.json())
            st.rerun()
        else:
            try:
                message = response.json()["detail"]
            except ValueError:
                message = response.text
            st.error(message)

    runs = requests.get(f"{api_url}/runs").json()
    run_names = [run["name"] for run in runs]

    if not run_names:
        st.info("Generate a run to see evaluation results.")
    else:
        selected_run = st.selectbox("Run", run_names)
        run = requests.get(f"{api_url}/runs/{selected_run}").json()
        predictions = pd.DataFrame(run["predictions"])
        metrics = pd.DataFrame(run["metrics"])

        if st.button("Calculate RAGAS metrics"):
            with st.spinner("RAGAS is evaluating the answers..."):
                response = requests.post(
                    f"{api_url}/metrics",
                    json={
                        "predictions_path": (
                            f"runs/{selected_run}/predictions.jsonl"
                        ),
                        "output_path": f"runs/{selected_run}/metrics.csv",
                    },
                )
            if response.ok:
                st.success(response.json())
                st.rerun()
            else:
                try:
                    message = response.json()["detail"]
                except ValueError:
                    message = response.text
                st.error(message)

        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        if not metrics.empty:
            for metric_name in metric_names:
                if metric_name not in metrics.columns:
                    metrics[metric_name] = pd.NA

            metric_columns = st.columns(4)
            for column, metric_name in zip(metric_columns, metric_names):
                values = pd.to_numeric(metrics[metric_name], errors="coerce")
                label = metric_name.replace("_", " ").title()
                column.metric(
                    label,
                    f"{values.mean():.3f}" if values.notna().any() else "N/A",
                )
                column.caption(
                    f"{values.notna().sum()}/{len(values)} scores completed"
                )

            st.subheader("Scores by question")
            if "id" not in metrics.columns:
                metrics["id"] = predictions["id"]
            st.dataframe(
                metrics[["id"] + metric_names],
                use_container_width=True,
            )
        else:
            st.warning("This run has predictions but no RAGAS metrics yet.")

        if not predictions.empty:
            st.subheader("Inspect a question")
            selected_id = st.selectbox(
                "Question ID",
                predictions["id"].tolist(),
            )
            result = predictions[predictions["id"] == selected_id].iloc[0]

            st.write("**Question:**", result["query"])
            st.write("**Expected answer:**", result["expected_answer"])
            st.write("**Generated answer:**", result["answer"])
            st.write("**Latency:**", f"{result['latency_ms']} ms")
            st.write("**Retrieved contexts:**")

            for number, context in enumerate(result["contexts"], 1):
                with st.expander(
                    f"Context {number}: {context['source_id']}"
                ):
                    st.write(context["text"])
                    st.caption(context["location"])


with qdrant_tab:
    st.header("Search Qdrant")
    query = st.text_input("Search the indexed documents")
    result_limit = st.number_input(
        "Number of results",
        min_value=1,
        max_value=20,
        value=5,
    )

    if st.button("Search Qdrant", disabled=not query):
        response = requests.post(
            f"{api_url}/search",
            json={"query": query, "limit": result_limit},
        )
        if response.ok:
            st.session_state["search_results"] = response.json()
        else:
            st.error(response.text)

    for number, result in enumerate(
        st.session_state.get("search_results", []),
        1,
    ):
        title = (
            f"{number}. {result['source_document']} "
            f"— score {result['score']:.3f}"
        )
        with st.expander(title):
            st.write("Search queries:", result.get("search_queries", []))
            st.write("Modality:", result["modality"])
            st.write("Location:", result["source_location"])
            st.write("Vector rank:", result.get("vector_rank", "Not in top 15"))
            st.write("BM25 rank:", result.get("bm25_rank", "Not in top 15"))
            st.write("Reranker score:", result.get("rerank_score"))
            if result["modality"] == "text":
                st.write(result["text"])
            else:
                st.image(result["image_path"])

    st.divider()
    st.header("Ingest document")
    uploaded_file = st.file_uploader(
        "PDF, Markdown, TXT, DOCX, PPTX or image",
        type=[
            "pdf",
            "md",
            "txt",
            "docx",
            "pptx",
            "png",
            "jpg",
            "jpeg",
            "webp",
            "tiff",
        ],
    )

    if st.button("Ingest document", disabled=uploaded_file is None):
        with st.spinner("Parsing, chunking, embedding and indexing document..."):
            response = requests.post(
                f"{api_url}/ingest",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                },
            )
        if response.ok:
            st.json(response.json())
        else:
            st.error(response.text)
