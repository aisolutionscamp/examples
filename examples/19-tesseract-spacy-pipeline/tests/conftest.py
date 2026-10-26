import pytest

INVOICE_TEXT = """Example Corp
1 Example Street, Springfield
Invoice No: INV-2026-0042
Invoice date: 2026-10-01
Due date: 31/10/2026
Consulting services 10 x 100.00 1,000.00 EUR
Subtotal: 1,000.00 EUR
VAT 19%: 190.00 EUR
Total: 1,190.00 EUR
Questions: billing@example.com"""

CONTRACT_TEXT = """SERVICE AGREEMENT
This Agreement is made between Example Corp and Sample Supplier GmbH, effective as of November 1, 2026.
The term runs until October 31, 2027. Termination requires three months notice.
Governing law and signatures of both parties follow."""


def make_text_pdf(text: str) -> bytes:
    """Minimal one-page PDF with a real text layer (Helvetica), built by hand."""
    lines = text.splitlines()
    ops = ["BT", "/F1 10 Tf", "12 TL", "40 800 Td"]
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({safe}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>"),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


@pytest.fixture
def invoice_pdf(tmp_path):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(make_text_pdf(INVOICE_TEXT))
    return path
