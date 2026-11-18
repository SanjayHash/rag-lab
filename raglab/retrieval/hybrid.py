
def reciprocal_rank_fusion(
    result_lists,
    k=60,
    top_k=3
):
    """
    Combine multiple ranked retrieval result lists
    using Reciprocal Rank Fusion.
    """

    if not result_lists:
        raise ValueError("Result lists cannot be empty")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    fused_scores = {}
    documents = {}

    for results in result_lists:

        for rank, result in enumerate(results, start=1):

            document = result["document"]

            # Use the chunk text as its unique identifier
            document_id = document["text"]

            score = 1 / (k + rank)

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0) + score
            )

            documents[document_id] = document

    ranked_documents = sorted(
        fused_scores,
        key=lambda document_id: fused_scores[document_id],
        reverse=True
    )

    fused_results = []

    for document_id in ranked_documents[:top_k]:

        fused_results.append({
            "document": documents[document_id],
            "score": fused_scores[document_id]
        })

    return fused_results
