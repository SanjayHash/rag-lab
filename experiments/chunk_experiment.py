
from raglab.chunking.fixed import fixed_size_chunk


sample_text = """
Machine learning is a subset of artificial intelligence.
It enables systems to learn patterns from data.

Supervised learning uses labelled datasets.
Unsupervised learning discovers patterns in unlabelled data.

Deep learning is a specialized branch of machine learning.
It uses neural networks with multiple layers.
"""

documents = [
    {
        "text": sample_text.strip(),
        "metadata": {
            "source": "machine_learning.pdf",
            "page": 1
        }
    }
]


experiments = [
    {"chunk_size": 50, "overlap": 10},
    {"chunk_size": 100, "overlap": 20},
    {"chunk_size": 200, "overlap": 40},
]


for config in experiments:

    chunks = fixed_size_chunk(
        documents,
        chunk_size=config["chunk_size"],
        overlap=config["overlap"]
    )

    print("\n" + "=" * 60)
    print("CHUNK SIZE:", config["chunk_size"])
    print("OVERLAP:", config["overlap"])
    print("TOTAL CHUNKS:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):

        print(f"\nChunk {index}:")
        print(chunk["text"])
