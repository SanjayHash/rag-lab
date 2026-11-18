
import streamlit as st
import tempfile
import os
import time
import hashlib

from raglab.pipeline.retrieval_pipeline import RetrievalPipeline
from raglab.generation.generator import AnswerGenerator


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Lab",
    page_icon="📚",
    layout="wide",
    
)

st.title("📚 RAG Lab")
st.caption("Interactive PDF Retrieval-Augmented Generation Experiment")

st.divider()


# --------------------------------------------------
# 2. Session state
# --------------------------------------------------

defaults = {
    "pipeline": None,
    "generator": None,
    "indexed_file": None,
    "indexed_file_hash": None,
    "indexed_config": None,
    "index_stats": None,
    "comparison_results": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 3. Sidebar settings
# --------------------------------------------------

st.sidebar.header("Retrieval Settings")

retrieval_mode = st.sidebar.selectbox(
    "Retrieval Strategy",
    ["dense", "hybrid", "hybrid_rerank"],
    format_func=lambda mode: {
        "dense": "Dense Retrieval",
        "hybrid": "Hybrid Retrieval",
        "hybrid_rerank": "Hybrid + Reranking"
    }[mode]
)

st.sidebar.caption(
    "Dense uses semantic similarity. Hybrid combines "
    "semantic and keyword search. Reranking adds a "
    "cross-encoder to reorder the candidates."
)

chunk_size = st.sidebar.slider(
    "Chunk Size (characters)",
    min_value=200,
    max_value=1000,
    value=500,
    step=100
)

overlap = st.sidebar.slider(
    "Chunk Overlap (characters)",
    min_value=0,
    max_value=300,
    value=100,
    step=50
)

top_k = st.sidebar.slider(
    "Final Retrieved Chunks",
    min_value=1,
    max_value=10,
    value=5
)

candidate_k = 10

if retrieval_mode != "dense":
    candidate_k = st.sidebar.slider(
        "Candidate Count",
        min_value=5,
        max_value=20,
        value=10
    )

current_config = {
    "retrieval_mode": retrieval_mode,
    "chunk_size": chunk_size,
    "overlap": overlap,
    "top_k": top_k,
    "candidate_k": candidate_k
}


# --------------------------------------------------
# 4. PDF upload
# --------------------------------------------------

st.header("1. Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)


# --------------------------------------------------
# 5. Index PDF
# --------------------------------------------------

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    st.info(f"Selected document: {uploaded_file.name}")

    if st.button("Index PDF", type="primary"):

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(file_bytes)
                temp_path = temp_file.name

            with st.spinner("Indexing PDF and loading models..."):

                start_time = time.perf_counter()

                pipeline = RetrievalPipeline(
                    chunk_size=chunk_size,
                    overlap=overlap,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                    candidate_k=candidate_k
                )

                stats = pipeline.index_pdf(temp_path)

                indexing_time = (
                    time.perf_counter() - start_time
                )

                # Preserve the original uploaded filename.
                for document in pipeline.documents:
                    document["metadata"]["source"] = (
                        uploaded_file.name
                    )

                generator = AnswerGenerator(
                    model_name="qwen3.5:2b"
                )

                st.session_state.pipeline = pipeline
                st.session_state.generator = generator
                st.session_state.indexed_file = uploaded_file.name
                st.session_state.indexed_file_hash = current_file_hash
                st.session_state.indexed_config = current_config.copy()
                st.session_state.index_stats = {
                    **stats,
                    "indexing_time": indexing_time
                }

            st.success("PDF indexed successfully!")

        except Exception as error:
            st.error(f"Indexing failed: {error}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# --------------------------------------------------
# 6. Display indexing statistics
# --------------------------------------------------

if st.session_state.index_stats is not None:

    stats = st.session_state.index_stats

    st.subheader("Indexing Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Pages Extracted",
        stats["pages_extracted"]
    )

    col2.metric(
        "Chunks Created",
        stats["chunks_created"]
    )

    col3.metric(
        "Embedding Dimensions",
        stats["embedding_shape"][1]
    )

    col4.metric(
        "Indexing Time",
        f"{stats['indexing_time']:.2f}s"
    )


# --------------------------------------------------
# 7. Question answering
# --------------------------------------------------

st.divider()

st.header("2. Ask a Question")

if st.session_state.pipeline is not None:

    st.caption(
        f"Currently indexed: {st.session_state.indexed_file}"
    )

    # Verify that the uploaded document and settings
    # still match the indexed configuration.
    same_file = (
        uploaded_file is not None
        and current_file_hash
        == st.session_state.indexed_file_hash
    )

    same_config = (
        current_config
        == st.session_state.indexed_config
    )

    ready = same_file and same_config

    if not same_file:
        st.warning(
            "The selected PDF differs from the indexed document. "
            "Please index it again."
        )

    elif not same_config:
        st.warning(
            "Retrieval settings have changed. "
            "Please re-index the PDF to apply them."
        )

    question = st.text_input(
        "Enter your question about the PDF"
    )

    if st.button(
        "Generate Answer",
        type="primary",
        disabled=not ready
    ):

        if not question.strip():
            st.warning("Please enter a question.")

        else:

            try:
                # Retrieval timing
                with st.spinner("Retrieving relevant chunks..."):

                    start_time = time.perf_counter()

                    retrieved_chunks = (
                        st.session_state.pipeline.search(question)
                    )

                    retrieval_time = (
                        time.perf_counter() - start_time
                    )

                # Generation timing
                with st.spinner("Generating answer using Qwen..."):

                    start_time = time.perf_counter()

                    result = st.session_state.generator.generate(
                        question=question,
                        retrieved_chunks=retrieved_chunks
                    )

                    generation_time = (
                        time.perf_counter() - start_time
                    )

                # Answer
                st.divider()
                st.subheader("Answer")

                st.markdown(result["answer"])

                # Timing metrics
                st.subheader("Performance")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Retrieval Time",
                    f"{retrieval_time:.2f}s"
                )

                col2.metric(
                    "Generation Time",
                    f"{generation_time:.2f}s"
                )

                col3.metric(
                    "Total Query Time",
                    f"{retrieval_time + generation_time:.2f}s"
                )

                # Sources
                st.subheader("Sources")

                for source in result["sources"]:
                    st.write(
                        f"- {source['source']} | "
                        f"Page {source['page']}"
                    )

                # Retrieved chunks
                with st.expander("Inspect Retrieved Chunks"):

                    for rank, item in enumerate(
                        retrieved_chunks,
                        start=1
                    ):

                        document = item["document"]

                        st.markdown(
                            f"### Rank {rank}"
                        )

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Relevance Score",
                            f"{item['score']:.5f}"
                        )

                        col2.metric(
                            "Source Page",
                            document["metadata"]["page"]
                        )

                        st.caption(
                            document["metadata"]["source"]
                        )

                        st.write(document["text"])

                        st.divider()

            except Exception as error:
                st.error(
                    f"Question answering failed: {error}"
                )

else:
    st.info(
        "Upload and index a PDF before asking questions."
    )


# --------------------------------------------------
# 8. Retrieval Strategy Comparison
# --------------------------------------------------

st.divider()

st.header("3. Compare Retrieval Strategies")

st.write(
    "Compare Dense, Hybrid and Hybrid + Reranking "
    "using the same indexed PDF and question."
)

if st.session_state.pipeline is not None:

    comparison_question = st.text_input(
        "Question for comparison",
        key="comparison_question"
    )

    if st.button("Compare Strategies", type="primary"):

        if not comparison_question.strip():

            st.warning("Please enter a question.")

        elif (
            uploaded_file is None
            or current_file_hash != st.session_state.indexed_file_hash
        ):

            st.warning("Please index the currently selected PDF first.")

        else:

            pipeline = st.session_state.pipeline

            strategies = [
                ("Dense Retrieval", "dense"),
                ("Hybrid Retrieval", "hybrid"),
                ("Hybrid + Reranking", "hybrid_rerank")
            ]

            comparison_results = []

            try:

                progress = st.progress(0)

                for index, (name, mode) in enumerate(strategies):

                    with st.spinner(f"Running {name}..."):

                        start_time = time.perf_counter()

                        results = pipeline.search(
                            comparison_question,
                            top_k=top_k,
                            retrieval_mode=mode
                        )

                        elapsed_time = (
                            time.perf_counter() - start_time
                        )

                    comparison_results.append({
                        "name": name,
                        "mode": mode,
                        "time": elapsed_time,
                        "results": results
                    })

                    progress.progress(
                        (index + 1) / len(strategies)
                    )

                st.session_state.comparison_results = (
                    comparison_results
                )

                st.success("Comparison completed!")

            except Exception as error:

                st.error(f"Comparison failed: {error}")


# --------------------------------------------------
# 9. Display comparison results
# --------------------------------------------------

if st.session_state.get("comparison_results"):

    comparison_results = (
        st.session_state.comparison_results
    )

    st.subheader("Performance Comparison")

    col1, col2, col3 = st.columns(3)

    for column, result in zip(
        [col1, col2, col3],
        comparison_results
    ):

        column.metric(
            result["name"],
            f"{result['time']:.2f}s"
        )

    st.subheader("Retrieved Documents")

    tabs = st.tabs([
        result["name"]
        for result in comparison_results
    ])

    for tab, result in zip(tabs, comparison_results):

        with tab:

            st.write(
                f"Retrieval time: {result['time']:.2f}s"
            )

            for rank, item in enumerate(
                result["results"],
                start=1
            ):

                document = item["document"]

                st.markdown(f"### Rank {rank}")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Relevance Score",
                    f"{item['score']:.5f}"
                )

                col2.metric(
                    "Page",
                    document["metadata"]["page"]
                )

                st.write(document["text"])

                st.divider()

