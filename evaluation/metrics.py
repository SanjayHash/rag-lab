
def precision_at_k(retrieved_pages, relevant_pages, k=5):
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_pages:
        raise ValueError("Relevant pages cannot be empty")

    retrieved = retrieved_pages[:k]

    relevant_count = sum(
        1 for page in retrieved
        if page in relevant_pages
    )

    return relevant_count / k


def recall_at_k(retrieved_pages, relevant_pages, k=5):
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_pages:
        raise ValueError("Relevant pages cannot be empty")

    retrieved = retrieved_pages[:k]

    relevant_count = sum(
        1 for page in retrieved
        if page in relevant_pages
    )

    return relevant_count / len(relevant_pages)


def reciprocal_rank(retrieved_pages, relevant_pages):
    if not relevant_pages:
        raise ValueError("Relevant pages cannot be empty")

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return 1 / rank

    return 0.0


def mean_reciprocal_rank(all_retrieved_pages, all_relevant_pages):
    if len(all_retrieved_pages) != len(all_relevant_pages):
        raise ValueError(
            "Each question must have corresponding relevant pages"
        )

    if not all_retrieved_pages:
        raise ValueError("Evaluation data cannot be empty")

    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(
            all_retrieved_pages,
            all_relevant_pages
        )
    ]

    return sum(scores) / len(scores)
