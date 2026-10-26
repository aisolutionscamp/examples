# Open-source document pipeline: Tesseract + spaCy

Self-hosted document processing without cloud services. A FastAPI service accepts PDFs and images, gets the text (PDF text layer or **Tesseract** OCR), classifies the document type with **scikit-learn**, extracts fields with **spaCy** rules and checks them.

## How it works

```
upload (PDF, PNG, JPG, TIFF, BMP; max 20 MB)
  └─► text.py      PDF text layer (pypdf) ─ too little text? ─► OCR of the page images (Tesseract)
                   images: grayscale, autocontrast, median filter, binarize ─► Tesseract (--oem 3 --psm 6)
  └─► classify.py  TF-IDF + logistic regression: invoice | contract | receipt | report | unknown
  └─► entities.py  spaCy blank "en" + EntityRuler: DATE, MONEY, EMAIL
                   fields from the words around each entity (Total, VAT, due, effective, between ... and ...)
  └─► validate.py  required fields, subtotal + tax = total, due date after invoice date
  └─► JSON response
```

- **No model downloads**: spaCy runs as a blank English pipeline with pattern rules, so it works offline and in tests. Set `SPACY_MODEL=en_core_web_sm` (after `python -m spacy download en_core_web_sm`) to add statistical NER; organization names then appear under `organizations`.
- **Dates** are normalized to ISO; slash and dot dates are read day-first (`31/10/2026`). Change `parse_date(day_first=False)` for US documents.
- **Amounts** understand `1,234.56` and `1.234,56` with a currency symbol or code next to them.
- **Classifier**: the bundled seed examples are only there so the pipeline runs. Train `train_classifier()` on a few hundred of your own labelled documents.

## Run

With Docker (includes the Tesseract binary):

```bash
docker build -t docproc examples/19-tesseract-spacy-pipeline
docker run --rm -p 8000:8000 docproc
curl -F file=@invoice.pdf localhost:8000/documents
```

Without Docker: install `tesseract-ocr` with your package manager, then `pip install -r requirements.txt` and `uvicorn docproc.api:app --app-dir src`. Text PDFs work even without Tesseract; images then return HTTP 503.

## Tests (offline)

```bash
scripts/test-example.sh examples/19-tesseract-spacy-pipeline   # from the repo root, runs in Docker
```

The tests build a real text PDF in memory, mock `pytesseract` for the image path, and check classification, field extraction, validation and the API (upload, wrong file type 415, broken file 422, missing OCR 503). One extra test renders an invoice as an image and runs the real Tesseract; it is skipped when the binary is not installed (as in `test-example.sh`) and passes with `tesseract-ocr` 5.5 installed.

## Limitations

- Rules are written for English invoices and contracts with labelled fields ("Total:", "Due date:"). Other layouts and languages need more patterns or a trained NER model.
- No table extraction: line items are not split into rows. For tables, add a layout tool (for example Camelot or pdfplumber for text PDFs) or use a layout-aware model.
- Tesseract quality drops on low-resolution scans (below about 300 dpi), photos at an angle and handwriting.
- Scanned PDFs are OCRed from their embedded images; PDFs that draw vector text outlines without a text layer need rasterization (for example with `pdf2image` and poppler).

## For production

- Measure field accuracy on a labelled sample of your own documents before trusting the `valid` status, and send `requires_review` / `invalid` documents to a person.
- Put the API behind authentication, scan uploads for malware, and run several workers (`uvicorn --workers`) or a queue for large batches.
- Install the Tesseract language packs you need (`tesseract-ocr-deu`, ...) and pass `lang` to `process_document`.
