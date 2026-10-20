from io import BytesIO

import pytest
from fastapi import HTTPException
from pypdf import PdfWriter

from main import extract_pdf_text


def make_blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_rejects_non_pdf_payload() -> None:
    with pytest.raises(HTTPException) as error:
        extract_pdf_text(b"not a PDF")
    assert error.value.status_code == 400


def test_ocr_text_is_used_for_pages_without_text_layer(monkeypatch) -> None:
    monkeypatch.setattr(
        "main.pytesseract.image_to_string",
        lambda image, lang: "OCR result",
    )

    pdf_bytes = make_blank_pdf()

    text, page_count = extract_pdf_text(pdf_bytes)

    assert text == "OCR result"
    assert page_count == 1


def test_extract_pdf_text_reports_page_progress(monkeypatch) -> None:
    monkeypatch.setattr(
        "main.extract_pdf_pages",
        lambda pdf_bytes, callback: (callback(1, 1), [(1, "PDF text")])[1],
    )
    progress = []

    text, page_count = extract_pdf_text(
        b"%PDF-test", lambda completed, total: progress.append((completed, total))
    )

    assert text == "PDF text"
    assert page_count == 1
    assert progress == [(1, 1)]


def test_rejects_pdf_without_text_after_ocr(monkeypatch) -> None:
    monkeypatch.setattr(
        "main.pytesseract.image_to_string",
        lambda image, lang: "",
    )

    with pytest.raises(HTTPException) as error:
        extract_pdf_text(make_blank_pdf())

    assert error.value.status_code == 422


def test_reports_missing_tesseract_for_scanned_pdf(monkeypatch) -> None:
    from pytesseract import TesseractNotFoundError

    def missing_tesseract(image, lang):
        raise TesseractNotFoundError()

    monkeypatch.setattr("main.pytesseract.image_to_string", missing_tesseract)

    with pytest.raises(HTTPException) as error:
        extract_pdf_text(make_blank_pdf())

    assert error.value.status_code == 503
    assert "Tesseract" in error.value.detail
