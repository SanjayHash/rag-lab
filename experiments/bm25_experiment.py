
from raglab.ingestion.pdf_loader import load_pdf
from raglab.chunking.recursive import recursive_chunk
from raglab.retrieval.bm25 import BM25Retriever
from raglab.embeddings.embedder import TextEmbedder
from raglab.retrieval.dense import dense_search


# 1. Load and chunk the PDF
pdf_path = "data/sample.pdf"

documents = load_pdf(pdf_path)

chunks = recursive_chunk(
    documents,
    chunk_size=500,
    overlap=100
)

print("Pages extracted:", len(documents))
print("Chunks created:", len(chunks))


# 2. Build BM25 index
bm25 = BM25Retriever()
bm25.index(chunks)


# 3. Build dense embeddings
embedder = TextEmbedder()

chunk_texts = [
    chunk["text"] for chunk in chunks
]

document_embeddings = embedder.embed_documents(chunk_texts)


# 4. Ask a question
query = input("\nEnter your question: ")

query_embedding = embedder.embed_query(query)


# 5. Retrieve using BM25
bm25_results = bm25.search(query, top_k=3)


# 6. Retrieve using dense search
dense_results = dense_search(
    query_embedding=query_embedding,
    document_embeddings=document_embeddings,
    documents=chunks,
    top_k=3
)


# 7. Display BM25 results
print("\n" + "=" * 60)
print("BM25 RESULTS")
print("=" * 60)

for rank, result in enumerate(bm25_results, start=1):

    document = result["document"]

    print(f"\nRank: {rank}")
    print("BM25 score:", round(result["score"], 4))
    print("Page:", document["metadata"]["page"])
    print("Text:", document["text"][:250])


# 8. Display dense results
print("\n" + "=" * 60)
print("DENSE RETRIEVAL RESULTS")
print("=" * 60)

for rank, result in enumerate(dense_results, start=1):

    document = result["document"]

    print(f"\nRank: {rank}")
    print("Cosine similarity:", round(result["score"], 4))
    print("Page:", document["metadata"]["page"])
    print("Text:", document["text"][:250])
