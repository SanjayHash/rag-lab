
from types import SimpleNamespace

import pytest

from raglab.generation.generator import AnswerGenerator


def test_generator_returns_answer_and_sources(monkeypatch):

    def mock_chat(**kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content="RNNs struggle with long-term dependencies. [Source 1]"
            )
        )

    monkeypatch.setattr(
        "raglab.generation.generator.ollama.chat",
        mock_chat
    )

    generator = AnswerGenerator()

    retrieved_chunks = [
        {
            "document": {
                "text": "RNNs process sequential information.",
                "metadata": {
                    "source": "sample.pdf",
                    "page": 31
                }
            },
            "score": 0.85
        }
    ]

    result = generator.generate(
        "Why do RNNs struggle?",
        retrieved_chunks
    )

    assert "RNNs struggle" in result["answer"]
    assert result["sources"] == [
        {"source": "sample.pdf", "page": 31}
    ]


def test_generator_removes_duplicate_sources(monkeypatch):

    def mock_chat(**kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(content="Test answer.")
        )

    monkeypatch.setattr(
        "raglab.generation.generator.ollama.chat",
        mock_chat
    )

    generator = AnswerGenerator()

    retrieved_chunks = [
        {
            "document": {
                "text": "First chunk.",
                "metadata": {"source": "sample.pdf", "page": 31}
            },
            "score": 0.9
        },
        {
            "document": {
                "text": "Second chunk.",
                "metadata": {"source": "sample.pdf", "page": 31}
            },
            "score": 0.8
        }
    ]

    result = generator.generate("Test question", retrieved_chunks)

    assert len(result["sources"]) == 1


def test_generator_rejects_empty_question():

    generator = AnswerGenerator()

    with pytest.raises(ValueError):
        generator.generate("", [{"document": {}}])


def test_generator_rejects_empty_retrieved_chunks():

    generator = AnswerGenerator()

    with pytest.raises(ValueError):
        generator.generate("Test question", [])
