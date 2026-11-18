
from raglab.chunking.fixed import fixed_size_chunk


def test_fixed_size_chunk():
    documents = [
        {
            "text": "A" * 1000,
            "metadata": {
                "source": "sample.pdf",
                "page": 1
            }
        }
    ]

    chunks = fixed_size_chunk(
        documents,
        chunk_size=500,
        overlap=100
    )

    assert len(chunks) == 3

    assert len(chunks[0]["text"]) == 500
    assert len(chunks[1]["text"]) == 500
    assert len(chunks[2]["text"]) == 200

    assert chunks[0]["text"][-100:] == chunks[1]["text"][:100]

    assert chunks[0]["metadata"]["source"] == "sample.pdf"
    assert chunks[1]["metadata"]["page"] == 1


def test_invalid_overlap():
    documents = [{"text": "Hello world", "metadata": {}}]

    try:
        fixed_size_chunk(documents, chunk_size=100, overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        pass
