# Simple invoice data extractor (OCR + regex + Excel)

A small command-line tool: point it at a folder of invoices and receipts (images, PDFs with text, or text files) and get one Excel sheet with vendor, invoice number, dates, subtotal, tax, total and currency. Rows that need a human check are highlighted.

## How it works

```
folder ─► ocr.py    .txt as is | PDF text layer (pypdf) | image ─► grayscale, 2x upscale, contrast ─► Tesseract
       ─► parse.py  regular expressions for number, dates, subtotal / tax / total, currency; plausibility checks
       ─► export.py Excel (openpyxl): bold header, date and number formats, filter, yellow rows = check
```

- **Dates**: `2026-10-03`, `03/10/2026`, `03.10.2026`, `03.10.26`. Slash and dot dates are read day-first by default; use `--us-dates` for month-first.
- **Amounts**: `1,234.56`, `1.234,56`, `4,00`, `$ 99.90`, `1 200,00`. The last two digits after `.` or `,` are the cents.
- **Labels**: total = `Total`, `Total Due`, `Amount due`, `Balance due`, `Grand total`; tax = `Tax`, `VAT`, `Sales tax`; subtotal = `Subtotal`, `Net`. The vendor is the first line of the document.
- **Checks** (column *Problems*): invoice number, date or total not found; subtotal + tax does not match total; due date before invoice date; unreadable files.

## Run

Install Tesseract first (`apt-get install tesseract-ocr`, `brew install tesseract`, or the Windows installer from the Tesseract project); it is only needed for images.

```bash
pip install -r requirements.txt
cd src
python -m invoice_extractor.cli ../samples -o invoices.xlsx
python -m invoice_extractor.cli scan.jpg --lang eng+deu --us-dates -o invoices.csv
```

Output:

```
example-corp.txt: INV-10442 1555.2 USD (ok)
sample-bakery-receipt.txt: 2026-0815 4.0 EUR (ok)
wrote invoices.xlsx
```

## Tests (offline)

```bash
scripts/test-example.sh examples/40-invoice-ocr-excel   # from the repo root, runs in Docker
```

The tests parse the two sample documents, check the amount and date parsers, mock Tesseract for the image path, and run the whole folder-to-Excel flow (text file, text PDF, PDF without text, ignored file type) and read the workbook back. One more test renders an invoice as an image and runs the real Tesseract; it is skipped when the binary is missing and passes with `tesseract-ocr` 5.5.

## Limitations

- Regular expressions only find what they are written for: labelled fields in English (plus `VAT`/`Net`). Unusual layouts, other languages or unlabelled totals need new patterns.
- Line items are not extracted.
- Scanned PDFs (no text layer) are reported, not OCRed. Convert them to images first, or use the Tesseract + spaCy pipeline or Azure AI Document Intelligence solution of the document-processing use case.
- OCR quality depends on the scan: 300 dpi, straight pages and good contrast work best; photos and handwriting do not.

## For production

- Collect 20 to 50 of your real invoices, check the Excel output by hand and adjust the patterns; keep those files as test cases.
- Keep the *Problems* column: never book an invoice whose row is highlighted without looking at the original.
- For many suppliers or layouts, move to a layout-aware service or model instead of growing the regex list.
