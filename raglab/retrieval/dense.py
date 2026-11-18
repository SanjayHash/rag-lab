
import numpy as np


def dense_search(
    query_embedding,
    document_embeddings,
    documents,
    top_k=3
):
    """
    Find the most relevant documents using cosine similarity.

    Assumes embeddings are already normalized.
    """

    query_embedding = np.asarray(query_embedding)
    document_embeddings = np.asarray(document_embeddings)

    if document_embeddings.ndim != 2:
        raise ValueError("Document embeddings must be a 2D matrix")

    if query_embedding.ndim != 1:
        raise ValueError("Query embedding must be a 1D vector")

    if len(documents) != len(document_embeddings):
        raise ValueError(
            "Number of documents must match number of embeddings"
        )

    if query_embedding.shape[0] != document_embeddings.shape[1]:
        raise ValueError("Query and document dimensions must match")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    # Calculate cosine similarity using dot product
    scores = np.dot(document_embeddings, query_embedding)

    # Get indices of the highest scores
    ranked_indices = np.argsort(scores)[::-1]

    # Return the top-k results
    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "document": documents[index],
            "score": float(scores[index])
        })

    return results
