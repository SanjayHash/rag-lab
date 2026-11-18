
from FlagEmbedding import FlagReranker


class DocumentReranker:

    def __init__(
        self,
        model_name="BAAI/bge-reranker-v2-m3",
        use_fp16=False
    ):
        """
        Load the cross-encoder reranking model.
        """

        self.reranker = FlagReranker(
            model_name,
            use_fp16=use_fp16,
            devices=["cpu"]
        )

    def rerank(self, query, retrieved_results, top_k=3):
        """
        Reorder retrieved chunks using cross-encoder scores.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not retrieved_results:
            raise ValueError("Retrieved results cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        pairs = []

        for result in retrieved_results:

            document = result["document"]

            pairs.append([
                query,
                document["text"]
            ])

        scores = self.reranker.compute_score(
            pairs,
            normalize=True
        )

        # Ensure scores can be processed as a list
        if not isinstance(scores, list):
            scores = [scores]

        reranked_results = []

        for result, score in zip(retrieved_results, scores):

            reranked_results.append({
                "document": result["document"],
                "score": float(score)
            })

        reranked_results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return reranked_results[:top_k]
