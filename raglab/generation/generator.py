
import ollama


class AnswerGenerator:

    def __init__(self, model_name="qwen3.5:2b"):
        self.model_name = model_name

    def generate(self, question, retrieved_chunks):

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if not retrieved_chunks:
            raise ValueError("No retrieved chunks provided")

        context_parts = []
        sources = []

        for index, result in enumerate(retrieved_chunks, start=1):

            document = result["document"]
            metadata = document["metadata"]

            page = metadata["page"]
            source = metadata["source"]
            text = document["text"]

            # Give every retrieved chunk a source identifier
            context_parts.append(
                f"[Source {index} | Page {page}]\n{text}"
            )

            # Preserve actual source metadata
            sources.append({
                "source": source,
                "page": page
            })

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a helpful assistant answering questions about a PDF.

Use only the provided context to answer the question.

Instructions:
- Do not invent information.
- If the context is insufficient, say so.
- Explain clearly and simply.
- Refer to supporting source identifiers such as [Source 1].

Context:
{context}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            think=False,
            options={
                "temperature": 0.2
            }
        )

        answer = response.message.content.strip()

        if not answer:
            raise RuntimeError("Qwen returned an empty answer")

        # Remove duplicate source references while preserving order
        unique_sources = []
        seen = set()

        for source in sources:

            key = (source["source"], source["page"])

            if key not in seen:
                seen.add(key)
                unique_sources.append(source)

        return {
            "answer": answer,
            "sources": unique_sources
        }
