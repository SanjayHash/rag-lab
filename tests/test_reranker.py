
import pytest

from raglab.retrieval.reranker import DocumentReranker


def create_test_reranker(scores):

    reranker = DocumentReranker.__new__(DocumentReranker)

    class MockReranker:

        def compute_score(self, pairs, normalize=True):
            return scores

    reranker.reranker = MockReranker()

    return reranker


def test_reranker_sorts_documents_by_score():

    reranker = create_test_reranker([0.3, 0.9, 0.5])

    results = [
        {"document": {"text": "Document A"}, "score": 0.8},
        {"document": {"text": "Document B"}, "score": 0.7},
        {"document": {"text": "Document C"}, "score": 0.6}
    ]

    reranked = reranker.rerank(
        "Test query",
        results,
        top_k=3
    )

    assert reranked[0]["document"]["text"] == "Document B"
    assert reranked[1]["document"]["text"] == "Document C"
    assert reranked[2]["document"]["text"] == "Document A"


def test_reranker_respects_top_k():

    reranker = create_test_reranker([0.3, 0.9, 0.5])

    results = [
        {"document": {"text": "Document A"}},
        {"document": {"text": "Document B"}},
        {"document": {"text": "Document C"}}
    ]

    reranked = reranker.rerank(
        "Test query",
        results,
        top_k=2
    )

    assert len(reranked) == 2


def test_reranker_rejects_empty_query():

    reranker = create_test_reranker([])

    with pytest.raises(ValueError):
        reranker.rerank("", [{"document": {"text": "Test"}}])


def test_reranker_rejects_empty_results():

    reranker = create_test_reranker([])

    with pytest.raises(ValueError):
        reranker.rerank("Test query", [])


def test_reranker_rejects_invalid_top_k():

    reranker = create_test_reranker([0.5])

    with pytest.raises(ValueError):
        reranker.rerank(
            "Test query",
            [{"document": {"text": "Test"}}],
            top_k=0
        )
