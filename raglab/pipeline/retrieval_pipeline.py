
from raglab.ingestion.pdf_loader import load_pdf
from raglab.chunking.recursive import recursive_chunk
from raglab.embeddings.embedder import TextEmbedder
from raglab.retrieval.dense import dense_search
from raglab.retrieval.bm25 import BM25Retriever
from raglab.retrieval.hybrid import reciprocal_rank_fusion
from raglab.retrieval.reranker import DocumentReranker


class RetrievalPipeline:

    def __init__(
        self,
        chunk_size=500,
        overlap=100,
        top_k=3,
        retrieval_mode="dense",
        candidate_k=10
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.candidate_k = candidate_k

        if retrieval_mode not in {
            "dense",
            "hybrid",
            "hybrid_rerank"
        }:
            raise ValueError("Invalid retrieval mode")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")

        self.embedder = TextEmbedder()

        self.documents = []
        self.document_embeddings = None

        self.bm25 = BM25Retriever()
        self.reranker = None

        if retrieval_mode == "hybrid_rerank":
            print("Loading cross-encoder reranker...")
            self.reranker = DocumentReranker()

    def index_pdf(self, file_path):

        documents = load_pdf(file_path)

        if not documents:
            raise ValueError("No extractable text found in PDF")

        chunks = recursive_chunk(
            documents,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )

        if not chunks:
            raise ValueError("No chunks were generated from PDF")

        chunk_texts = [
            chunk["text"] for chunk in chunks
        ]

        embeddings = self.embedder.embed_documents(chunk_texts)

        self.bm25.index(chunks)

        self.documents = chunks
        self.document_embeddings = embeddings

        return {
            "pages_extracted": len(documents),
            "chunks_created": len(chunks),
            "embedding_shape": embeddings.shape
        }

    def search(self, query, top_k=None, retrieval_mode=None):

        if self.document_embeddings is None:
            raise RuntimeError("Please index a PDF before searching")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        final_top_k = (
            top_k if top_k is not None else self.top_k
        )

        if final_top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Use the requested mode or the pipeline's default mode.
        mode = retrieval_mode or self.retrieval_mode

        if mode not in {
            "dense",
            "hybrid",
            "hybrid_rerank"
        }:
            raise ValueError("Invalid retrieval mode")

        query_embedding = self.embedder.embed_query(query)

        # Step 1: Dense retrieval
        dense_results = dense_search(
            query_embedding=query_embedding,
            document_embeddings=self.document_embeddings,
            documents=self.documents,
            top_k=(
                final_top_k
                if mode == "dense"
                else self.candidate_k
            )
        )

        if mode == "dense":
            return dense_results[:final_top_k]

        # Step 2: BM25 retrieval
        bm25_results = self.bm25.search(
            query,
            top_k=self.candidate_k
        )

        # Step 3: Hybrid retrieval
        hybrid_results = reciprocal_rank_fusion(
            [bm25_results, dense_results],
            k=60,
            top_k=self.candidate_k
        )

        if mode == "hybrid":
            return hybrid_results[:final_top_k]

        # Step 4: Cross-encoder reranking
        if self.reranker is None:
            print("Loading cross-encoder reranker...")
            self.reranker = DocumentReranker()

        return self.reranker.rerank(
            query=query,
            retrieved_results=hybrid_results,
            top_k=final_top_k
        )
