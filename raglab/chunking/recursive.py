
def recursive_chunk(
    documents,
    chunk_size=500,
    overlap=100
):
    """
    Split documents using increasingly smaller separators.

    Attempts to preserve paragraphs, sentences and words.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    separators = ["\n\n", "\n", ". ", " ", ""]

    chunks = []

    def split_text(text, separator_index=0):

        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        if separator_index >= len(separators):
            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        separator = separators[separator_index]

        if separator == "":
            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        parts = text.split(separator)

        if len(parts) == 1:
            return split_text(text, separator_index + 1)

        result = []
        current = ""

        for part in parts:

            candidate = current + separator + part if current else part

            if len(candidate) <= chunk_size:
                current = candidate

            else:
                if current:
                    result.extend(
                        split_text(current, separator_index + 1)
                    )

                current = part

        if current:
            result.extend(
                split_text(current, separator_index + 1)
            )

        return result

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        document_chunks = split_text(text)

        # Add overlap between consecutive chunks
        final_chunks = []

        for index, chunk in enumerate(document_chunks):

            if index > 0 and overlap > 0:
                previous = document_chunks[index - 1]
                prefix = previous[-overlap:]
                chunk = prefix + chunk

            final_chunks.append(chunk)

        for chunk in final_chunks:

            if chunk.strip():
                chunks.append({
                    "text": chunk.strip(),
                    "metadata": metadata.copy()
                })

    return chunks
