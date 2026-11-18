
import pytest

from raglab.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_combines_rankings():

    document_a = {"text": "Document A"}
    document_b = {"text": "Document B"}
    document_c = {"text": "Document C"}

    bm25_results = [
        {"document": document_a, "score": 8.0},
        {"document": document_b, "score": 6.0}
    ]

    dense_results = [
        {"document": document_b, "score": 0.9},
        {"document": document_c, "score": 0.8}
    ]

    results = reciprocal_rank_fusion(
        [bm25_results, dense_results],
        top_k=3
    )

    # Document B ranks well in both systems
    assert results[0]["document"]["text"] == "Document B"


def test_rrf_rejects_empty_result_lists():

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([])


def test_rrf_rejects_invalid_k():

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [[]],
            k=0
        )


def test_rrf_rejects_invalid_top_k():

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [[]],
            top_k=0
        )
