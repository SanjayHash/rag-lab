
from raglab.embeddings.embedder import TextEmbedder
from raglab.retrieval.dense import dense_search


embedder = TextEmbedder()

documents = [
    {
        "text": "Machine learning uses data to train models.",
        "metadata": {"page": 1}
    },
    {
        "text": "Neural networks are inspired by the human brain.",
        "metadata": {"page": 2}
    },
    {
        "text": "Cooking pasta requires boiling water.",
        "metadata": {"page": 3}
    },
]

# Generate document embeddings
document_texts = [
    document["text"] for document in documents
]

document_embeddings = embedder.embed_documents(document_texts)

# Generate query embedding
query = "How do machines learn from data?"

query_embedding = embedder.embed_query(query)

# Retrieve relevant documents
results = dense_search(
    query_embedding=query_embedding,
    document_embeddings=document_embeddings,
    documents=documents,
    top_k=2
)

print("\nQuery:", query)
print("\nRetrieved documents:")

for result in results:
    print("\nScore:", round(result["score"], 4))
    print("Text:", result["document"]["text"])
    print("Metadata:", result["document"]["metadata"])
