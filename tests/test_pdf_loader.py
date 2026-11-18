
import fitz

from raglab.ingestion.pdf_loader import load_pdf


def test_load_pdf(tmp_path):
    # Create a small sample PDF for testing
    pdf_path = tmp_path / "sample.pdf"

    pdf = fitz.open()

    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "Machine learning is a subset of artificial intelligence."
    )

    pdf.save(pdf_path)
    pdf.close()

    # Test our PDF loader
    documents = load_pdf(str(pdf_path))

    assert len(documents) == 1
    assert "Machine learning" in documents[0]["text"]
    assert documents[0]["metadata"]["page"] == 1
    assert documents[0]["metadata"]["source"] == "sample.pdf"
