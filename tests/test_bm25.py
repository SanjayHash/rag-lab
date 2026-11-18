
import pytest

from raglab.retrieval.bm25 import BM25Retriever


def test_bm25_returns_relevant_document_first():

    documents = [
        {"text": "Cooking pasta requires boiling water."},
        {"text": "Recurrent neural networks process sequential data."},
        {"text": "Neural networks are used in deep learning."}
    ]

    retriever = BM25Retriever()
    retriever.index(documents)

    results = retriever.search(
        "recurrent neural networks",
        top_k=2
    )

    assert len(results) == 2
    assert results[0]["document"]["text"] == (
        "Recurrent neural networks process sequential data."
    )


def test_bm25_rejects_empty_documents():

    retriever = BM25Retriever()

    with pytest.raises(ValueError):
        retriever.index([])


def test_bm25_rejects_search_before_indexing():

    retriever = BM25Retriever()

    with pytest.raises(RuntimeError):
        retriever.search("neural networks")


def test_bm25_rejects_empty_query():

    retriever = BM25Retriever()
    retriever.index([{"text": "Neural networks learn from data."}])

    with pytest.raises(ValueError):
        retriever.search("")


def test_bm25_rejects_invalid_top_k():

    retriever = BM25Retriever()
    retriever.index([{"text": "Neural networks learn from data."}])

    with pytest.raises(ValueError):
        retriever.search("neural networks", top_k=0)
