
from raglab.chunking.recursive import recursive_chunk


def test_recursive_chunk_preserves_metadata():

    documents = [
        {
            "text": (
                "Machine learning is a subset of artificial intelligence. "
                "It enables systems to learn from data.\n\n"
                "Deep learning uses neural networks with multiple layers."
            ),
            "metadata": {
                "source": "sample.pdf",
                "page": 1
            }
        }
    ]

    chunks = recursive_chunk(
        documents,
        chunk_size=80,
        overlap=10
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk["metadata"]["source"] == "sample.pdf"
        assert chunk["metadata"]["page"] == 1
        assert chunk["text"].strip()


def test_recursive_chunk_invalid_size():

    documents = [{"text": "Hello world", "metadata": {}}]

    try:
        recursive_chunk(documents, chunk_size=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass
