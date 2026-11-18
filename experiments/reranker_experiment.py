
import time

from raglab.ingestion.pdf_loader import load_pdf
from raglab.chunking.recursive import recursive_chunk
from raglab.embeddings.embedder import TextEmbedder
from raglab.retrieval.dense import dense_search
from raglab.retrieval.bm25 import BM25Retriever
from raglab.retrieval.hybrid import reciprocal_rank_fusion
from raglab.retrieval.reranker import DocumentReranker


# 1. Load and chunk PDF
print("Loading PDF...")

documents = load_pdf("data/sample.pdf")

chunks = recursive_chunk(
    documents,
    chunk_size=500,
    overlap=100
)

print("Pages extracted:", len(documents))
print("Chunks created:", len(chunks))


# 2. Initialize embedding model
print("\nLoading embedding model...")

embedder = TextEmbedder()

chunk_texts = [
    chunk["text"] for chunk in chunks
]

document_embeddings = embedder.embed_documents(chunk_texts)


# 3. Initialize BM25
bm25 = BM25Retriever()
bm25.index(chunks)


# 4. Get query
query = input("\nEnter your question: ")

query_embedding = embedder.embed_query(query)


# 5. Dense retrieval
dense_results = dense_search(
    query_embedding=query_embedding,
    document_embeddings=document_embeddings,
    documents=chunks,
    top_k=10
)


# 6. BM25 retrieval
bm25_results = bm25.search(
    query,
    top_k=10
)


# 7. Hybrid retrieval
hybrid_results = reciprocal_rank_fusion(
    [bm25_results, dense_results],
    k=60,
    top_k=10
)


# 8. Initialize reranker
print("\nLoading cross-encoder reranker...")
start_time = time.time()

reranker = DocumentReranker()

print(
    "Reranker loaded in:",
    round(time.time() - start_time, 2),
    "seconds"
)


# 9. Rerank hybrid results
print("\nReranking documents...")
start_time = time.time()

reranked_results = reranker.rerank(
    query=query,
    retrieved_results=hybrid_results,
    top_k=5
)

reranking_time = time.time() - start_time


# 10. Display results
def display_results(title, results):

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    for rank, result in enumerate(results, start=1):

        document = result["document"]

        print(f"\nRank: {rank}")
        print("Score:", round(result["score"], 5))
        print("Page:", document["metadata"]["page"])
        print("Text:", document["text"][:300])


display_results(
    "HYBRID RETRIEVAL",
    hybrid_results
)

display_results(
    "HYBRID + RERANKING",
    reranked_results
)

print("\nReranking time:", round(reranking_time, 2), "seconds")
