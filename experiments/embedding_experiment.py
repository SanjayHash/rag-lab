
import numpy as np

from raglab.embeddings.embedder import TextEmbedder


embedder = TextEmbedder()

sentences = [
    "Machine learning enables computers to learn from data.",
    "Artificial intelligence allows machines to perform intelligent tasks.",
    "I enjoy cooking pasta for dinner.",
]

# Convert sentences into embeddings
vectors = embedder.embed_documents(sentences)

print("Embedding matrix shape:", vectors.shape)
print("First vector shape:", vectors[0].shape)
print("First 10 values:", vectors[0][:10])

# Compare the first sentence with the other sentences
query_vector = embedder.embed_query(
    "How do computers learn from examples?"
)

similarities = np.dot(vectors, query_vector)

print("\nSimilarity scores:")

for sentence, score in zip(sentences, similarities):
    print(f"{score:.4f} - {sentence}")
