
from raglab.pipeline.retrieval_pipeline import RetrievalPipeline

from evaluation.retrieval_dataset import EVALUATION_DATASET

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank
)


PDF_PATH = "data/sample.pdf"
K = 5

STRATEGIES = [
    "dense",
    "hybrid",
    "hybrid_rerank"
]


def evaluate_strategy(pipeline, strategy):

    all_precision = []
    all_recall = []
    all_mrr = []

    for item in EVALUATION_DATASET:

        question = item["question"]
        relevant_pages = item["relevant_pages"]

        results = pipeline.search(
            question,
            top_k=K,
            retrieval_mode=strategy
        )
        retrieved_pages = []

        for result in results:

            page = result["document"]["metadata"]["page"]

            if page not in retrieved_pages:
                retrieved_pages.append(page)

        all_precision.append(
            precision_at_k(
                retrieved_pages,
                relevant_pages,
                K
            )
        )

        all_recall.append(
            recall_at_k(
                retrieved_pages,
                relevant_pages,
                K
            )
        )

        all_mrr.append(
            reciprocal_rank(
                retrieved_pages,
                relevant_pages
            )
        )

    return {
        "precision": sum(all_precision) / len(all_precision),
        "recall": sum(all_recall) / len(all_recall),
        "mrr": sum(all_mrr) / len(all_mrr)
    }


def main():

    print("Initializing RAG pipeline...")

    pipeline = RetrievalPipeline(
        chunk_size=500,
        overlap=100,
        top_k=K,
        retrieval_mode="dense",
        candidate_k=10
    )

    print("\nIndexing PDF...")

    stats = pipeline.index_pdf(PDF_PATH)

    print("Chunks created:", stats["chunks_created"])

    print("\nEvaluating retrieval strategies...\n")

    for strategy in STRATEGIES:

        print("=" * 50)
        print("Strategy:", strategy)

        metrics = evaluate_strategy(
            pipeline,
            strategy
        )

        print(f"Precision@{K}: {metrics['precision']:.4f}")
        print(f"Recall@{K}:    {metrics['recall']:.4f}")
        print(f"MRR:           {metrics['mrr']:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    main()
