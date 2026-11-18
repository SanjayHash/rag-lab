
from raglab.pipeline.retrieval_pipeline import RetrievalPipeline


pipeline = RetrievalPipeline(
    chunk_size=500,
    overlap=100,
    top_k=3
)

# Replace this with the path to your own PDF
pdf_path = "data/sample.pdf"

# Index the PDF
print("Indexing PDF...")

stats = pipeline.index_pdf(pdf_path)

print("\nIndexing complete!")
print("Pages extracted:", stats["pages_extracted"])
print("Chunks created:", stats["chunks_created"])
print("Embedding shape:", stats["embedding_shape"])


# Ask a question
query = input("\nEnter your question: ")

results = pipeline.search(query)

print("\nRetrieved chunks:")

for index, result in enumerate(results, start=1):

    document = result["document"]

    print(f"\nResult {index}")
    print("Similarity:", round(result["score"], 4))
    print("Page:", document["metadata"]["page"])
    print("Text:", document["text"])
