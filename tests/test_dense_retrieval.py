
import numpy as np
import pytest

from raglab.retrieval.dense import dense_search


def test_dense_search_returns_most_similar_documents():

    query = np.array([1.0, 0.0])

    document_embeddings = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.7, 0.7]
    ])

    documents = [
        {"text": "Unrelated document"},
        {"text": "Most relevant document"},
        {"text": "Partially relevant document"}
    ]

    results = dense_search(
        query,
        document_embeddings,
        documents,
        top_k=2
    )

    assert len(results) == 2

    assert (
        results[0]["document"]["text"]
        == "Most relevant document"
    )

    assert results[0]["score"] == pytest.approx(1.0)


def test_dense_search_rejects_dimension_mismatch():

    query = np.array([1.0, 0.0])

    document_embeddings = np.array([
        [1.0, 0.0, 0.0]
    ])

    documents = [{"text": "Sample"}]

    with pytest.raises(ValueError):
        dense_search(
            query,
            document_embeddings,
            documents
        )


def test_dense_search_rejects_invalid_top_k():

    query = np.array([1.0, 0.0])

    document_embeddings = np.array([
        [1.0, 0.0]
    ])

    documents = [{"text": "Sample"}]

    with pytest.raises(ValueError):
        dense_search(
            query,
            document_embeddings,
            documents,
            top_k=0
        )
