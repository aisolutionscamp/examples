"""Get text out of PDFs and images: the PDF text layer first, Tesseract OCR as fallback.

OCR needs the tesseract binary (apt-get install tesseract-ocr). Without it,
text PDFs still work and images raise OCRUnavailable.
"""
import io
import shutil
from pathlib import Path

import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pypdf import PdfReader

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MIN_TEXT_CHARS = 50  # less than this on a PDF page means "probably a scan"


class OCRUnavailable(RuntimeError):
    pass


def ocr_available() -> bool:
    return shutil.which(pytesseract.pytesseract.tesseract_cmd) is not None


def preprocess(image: Image.Image) -> Image.Image:
    """Grayscale, stretch contrast, remove speckles, binarize: helps Tesseract on scans."""
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    gray = gray.filter(ImageFilter.MedianFilter(3))
    return gray.point(lambda p: 255 if p > 160 else 0)


def ocr_image(image: Image.Image, lang: str = "eng") -> str:
    if not ocr_available():
        raise OCRUnavailable("tesseract binary not found; install tesseract-ocr or send a text PDF")
    # --psm 6: treat the page as one uniform block of text, works well for invoices
    return pytesseract.image_to_string(preprocess(image), lang=lang, config="--oem 3 --psm 6")


def extract_text(path: str | Path, lang: str = "eng") -> tuple[str, str]:
    """Return (text, method) where method is 'pdf-text', 'pdf-ocr' or 'image-ocr'."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        if sum(len(p.strip()) for p in pages) >= MIN_TEXT_CHARS * len(pages):
            return "\n".join(pages), "pdf-text"
        # scanned PDF: OCR the images embedded in each page
        texts = [ocr_image(Image.open(io.BytesIO(img.data)), lang)
                 for page in reader.pages for img in page.images]
        return "\n".join(texts), "pdf-ocr"
    if suffix in IMAGE_TYPES:
        with Image.open(path) as img:
            return ocr_image(img, lang), "image-ocr"
    raise ValueError(f"unsupported file type: {suffix}")
