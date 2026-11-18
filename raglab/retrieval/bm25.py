
import re

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):
        """
        Initialize an empty BM25 retriever.
        """
        self.documents = []
        self.tokenized_documents = []
        self.bm25 = None

    @staticmethod
    def tokenize(text):
        """
        Convert text into lowercase word tokens.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def index(self, documents):
        """
        Build the BM25 index from document chunks.
        """

        if not documents:
            raise ValueError("Documents cannot be empty")

        self.documents = documents

        self.tokenized_documents = [
            self.tokenize(document["text"])
            for document in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

    def search(self, query, top_k=3):
        """
        Retrieve the most relevant chunks using BM25.
        """

        if self.bm25 is None:
            raise RuntimeError("Please index documents before searching")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            results.append({
                "document": self.documents[index],
                "score": float(scores[index])
            })

        return results
