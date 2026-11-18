
from raglab.ingestion.pdf_loader import load_pdf
from raglab.chunking.recursive import recursive_chunk
from raglab.embeddings.embedder import TextEmbedder
from raglab.retrieval.dense import dense_search
from raglab.retrieval.bm25 import BM25Retriever
from raglab.retrieval.hybrid import reciprocal_rank_fusion


# 1. Load and chunk PDF
pdf_path = "data/sample.pdf"

documents = load_pdf(pdf_path)

chunks = recursive_chunk(
    documents,
    chunk_size=500,
    overlap=100
)

print("Pages extracted:", len(documents))
print("Chunks created:", len(chunks))


# 2. Initialize embedding model
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


# 5. Retrieve with BM25
bm25_results = bm25.search(
    query,
    top_k=5
)


# 6. Retrieve with dense search
dense_results = dense_search(
    query_embedding=query_embedding,
    document_embeddings=document_embeddings,
    documents=chunks,
    top_k=5
)


# 7. Combine rankings using RRF
hybrid_results = reciprocal_rank_fusion(
    [bm25_results, dense_results],
    k=60,
    top_k=5
)


# 8. Display results
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


display_results("BM25 RESULTS", bm25_results)

display_results("DENSE RETRIEVAL RESULTS", dense_results)

display_results("HYBRID RETRIEVAL RESULTS", hybrid_results)
