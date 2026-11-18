
import pymupdf
import os

def load_pdf(file_path):
    """
    Extract text from a PDF, page by page.

    Returns a list of dictionaries containing
    page text and metadata.
    """

    documents = []

    pdf = pymupdf.open(file_path)

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text("text")

        if not text.strip():
            continue

        document = {
            "text": text.strip(),
            "metadata": {
                "source": os.path.basename(file_path),
                "page": page_number
            }
        }

        documents.append(document)

    pdf.close()

    return documents
