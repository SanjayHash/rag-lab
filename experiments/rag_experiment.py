
from raglab.pipeline.retrieval_pipeline import RetrievalPipeline
from raglab.generation.generator import AnswerGenerator


# 1. Initialize the retrieval pipeline
pipeline = RetrievalPipeline(
    chunk_size=500,
    overlap=100,
    top_k=5,
    retrieval_mode="hybrid_rerank",
    candidate_k=10
)

# 2. Initialize the answer generator
generator = AnswerGenerator(
    model_name="qwen3.5:2b"
)

# 3. Index the PDF
pdf_path = "data/sample.pdf"

print("\nIndexing PDF...")

stats = pipeline.index_pdf(pdf_path)

print("\nIndexing complete!")
print("Pages extracted:", stats["pages_extracted"])
print("Chunks created:", stats["chunks_created"])
print("Embedding shape:", stats["embedding_shape"])

# 4. Get the user's question
question = input("\nAsk a question about your PDF: ")

# 5. Retrieve and rerank relevant chunks
print("\nRetrieving and reranking relevant chunks...")

retrieved_chunks = pipeline.search(question)

print("\nRetrieved chunks:")

for rank, result in enumerate(retrieved_chunks, start=1):

    document = result["document"]

    print(f"\nRank: {rank}")
    print("Reranker score:", round(result["score"], 5))
    print("Page:", document["metadata"]["page"])

# 6. Generate the answer
print("\nGenerating answer using Qwen...\n")

result = generator.generate(
    question=question,
    retrieved_chunks=retrieved_chunks
)

# 7. Display final answer
print("\n" + "=" * 65)

print("QUESTION:")
print(question)

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")

for source in result["sources"]:
    print(
        f"- {source['source']} | Page {source['page']}"
    )

print("=" * 65)
