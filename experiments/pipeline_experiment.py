
from raglab.pipeline.retrieval_pipeline import RetrievalPipeline


def display_results(title, results):

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    for rank, result in enumerate(results, start=1):

        document = result["document"]

        print(f"\nRank: {rank}")
        print("Score:", round(result["score"], 5))
        print("Page:", document["metadata"]["page"])
        print("Text:", document["text"][:250])


query = input("\nEnter your question: ")

# Test 1: Dense retrieval
dense_pipeline = RetrievalPipeline(
    retrieval_mode="dense",
    top_k=5
)

print("\nIndexing PDF for dense retrieval...")
dense_pipeline.index_pdf("data/sample.pdf")

dense_results = dense_pipeline.search(query)

display_results("DENSE RETRIEVAL", dense_results)


# Test 2: Hybrid retrieval
hybrid_pipeline = RetrievalPipeline(
    retrieval_mode="hybrid",
    top_k=5,
    candidate_k=10
)

print("\nIndexing PDF for hybrid retrieval...")
hybrid_pipeline.index_pdf("data/sample.pdf")

hybrid_results = hybrid_pipeline.search(query)

display_results("HYBRID RETRIEVAL", hybrid_results)


# Test 3: Hybrid + reranking
rerank_pipeline = RetrievalPipeline(
    retrieval_mode="hybrid_rerank",
    top_k=5,
    candidate_k=10
)

print("\nIndexing PDF for hybrid + reranking...")
rerank_pipeline.index_pdf("data/sample.pdf")

reranked_results = rerank_pipeline.search(query)

display_results("HYBRID + RERANKING", reranked_results)
