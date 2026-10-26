from datetime import date

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from docproc import text as text_mod
from docproc.api import app
from docproc.classify import classify, train_classifier
from docproc.entities import extract_fields, parse_date
from docproc.pipeline import process_document
from docproc.text import OCRUnavailable, extract_text, preprocess

from .conftest import CONTRACT_TEXT, INVOICE_TEXT, make_text_pdf


def test_text_pdf_is_read_without_ocr(invoice_pdf, monkeypatch):
    monkeypatch.setattr(text_mod, "ocr_image", lambda *a, **k: pytest.fail("OCR must not run"))
    text, method = extract_text(invoice_pdf)
    assert method == "pdf-text" and "INV-2026-0042" in text


def test_image_goes_through_preprocessing_and_tesseract(tmp_path, monkeypatch):
    img_path = tmp_path / "scan.png"
    Image.new("RGB", (60, 30), "white").save(img_path)
    seen = {}

    def fake_ocr(image, lang, config):
        seen.update(mode=image.mode, lang=lang, config=config)
        return INVOICE_TEXT

    monkeypatch.setattr(text_mod, "ocr_available", lambda: True)
    monkeypatch.setattr(text_mod.pytesseract, "image_to_string", fake_ocr)
    text, method = extract_text(img_path)
    assert method == "image-ocr" and text == INVOICE_TEXT
    assert seen == {"mode": "L", "lang": "eng", "config": "--oem 3 --psm 6"}


def test_missing_tesseract_gives_clear_error(tmp_path, monkeypatch):
    img_path = tmp_path / "scan.png"
    Image.new("RGB", (10, 10), "white").save(img_path)
    monkeypatch.setattr(text_mod, "ocr_available", lambda: False)
    with pytest.raises(OCRUnavailable):
        extract_text(img_path)
    with pytest.raises(ValueError):
        extract_text(tmp_path / "file.docx")


def test_preprocess_binarizes():
    out = preprocess(Image.new("RGB", (4, 4), (200, 200, 200)))
    assert out.mode == "L" and sum(out.histogram()[1:255]) == 0  # only black and white left


def test_classifier():
    assert classify(INVOICE_TEXT)[0] == "invoice"
    assert classify(CONTRACT_TEXT)[0] == "contract"
    custom = train_classifier(["alpha beta", "gamma delta"], ["a", "g"])
    assert classify("alpha", model=custom)[0] == "a"


def test_invoice_fields():
    f = extract_fields(INVOICE_TEXT, "invoice")
    assert f["vendor"] == "Example Corp"
    assert f["invoice_number"] == "INV-2026-0042"
    assert f["invoice_date"] == date(2026, 10, 1) and f["due_date"] == date(2026, 10, 31)
    assert (f["subtotal"], f["tax"], f["total"]) == (1000.0, 190.0, 1190.0)
    assert f["emails"] == ["billing@example.com"]


def test_contract_fields():
    f = extract_fields(CONTRACT_TEXT, "contract")
    assert f["parties"] == ["Example Corp", "Sample Supplier GmbH"]
    assert f["effective_date"] == date(2026, 11, 1) and f["end_date"] == date(2027, 10, 31)


@pytest.mark.parametrize("raw,expected", [("2026 - 10 - 01", date(2026, 10, 1)), ("31.10.2026", date(2026, 10, 31)),
                                          ("Sept 3, 2026", date(2026, 9, 3)), ("no date", None)])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_full_pipeline_marks_valid_invoice(invoice_pdf):
    result = process_document(invoice_pdf)
    assert result["document_type"] == "invoice"
    assert result["validation"]["errors"] == []


def test_api_upload_and_errors(tmp_path, monkeypatch):
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/documents", files={"file": ("../../etc/invoice.pdf", make_text_pdf(INVOICE_TEXT),
                                                  "application/pdf")})
    assert r.status_code == 200 and r.json()["fields"]["invoice_number"] == "INV-2026-0042"
    assert r.json()["fields"]["invoice_date"] == "2026-10-01"
    assert client.post("/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415
    assert client.post("/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}).status_code == 422
    monkeypatch.setattr(text_mod, "ocr_available", lambda: False)
    png = tmp_path / "s.png"
    Image.new("RGB", (10, 10), "white").save(png)
    assert client.post("/documents", files={"file": ("s.png", png.read_bytes(), "image/png")}).status_code == 503


@pytest.mark.skipif(not text_mod.ocr_available(), reason="tesseract binary not installed")
def test_real_ocr_on_rendered_invoice(tmp_path):
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (1400, 900), "white")
    ImageDraw.Draw(img).multiline_text((40, 40), INVOICE_TEXT, fill="black",
                                       font=ImageFont.load_default(size=32), spacing=12)
    img.save(tmp_path / "scan.png")
    result = process_document(tmp_path / "scan.png")
    assert result["text_method"] == "image-ocr"
    assert result["fields"]["invoice_number"] == "INV-2026-0042" and result["fields"]["total"] == 1190.0
