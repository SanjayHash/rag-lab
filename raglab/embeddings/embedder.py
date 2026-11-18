
from sentence_transformers import SentenceTransformer


class TextEmbedder:

    def __init__(
        self,
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu"
    ):
        """
        Load the embedding model.
        """

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

    def embed_documents(self, texts):
        """
        Convert multiple document chunks into embeddings.
        """

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def embed_query(self, query):
        """
        Convert a user question into an embedding.
        """

        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
