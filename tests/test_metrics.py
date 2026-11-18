
import pytest

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank
)


def test_precision_at_k():
    assert precision_at_k(
        [30, 31, 32, 22, 34],
        [31, 32],
        k=5
    ) == pytest.approx(0.4)


def test_recall_at_k():
    assert recall_at_k(
        [30, 31, 32, 22, 34],
        [31, 32],
        k=5
    ) == pytest.approx(1.0)


def test_reciprocal_rank():
    assert reciprocal_rank(
        [30, 31, 32],
        [31, 32]
    ) == pytest.approx(0.5)


def test_reciprocal_rank_when_no_relevant_result():
    assert reciprocal_rank(
        [30, 22, 34],
        [31, 32]
    ) == 0.0


def test_mean_reciprocal_rank():
    retrieved = [
        [30, 31, 32],
        [32, 30, 31]
    ]

    relevant = [
        [31, 32],
        [32]
    ]

    assert mean_reciprocal_rank(
        retrieved,
        relevant
    ) == pytest.approx(0.75)


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k([30, 31], [31], k=0)


def test_empty_relevant_pages():
    with pytest.raises(ValueError):
        recall_at_k([30, 31], [], k=2)
