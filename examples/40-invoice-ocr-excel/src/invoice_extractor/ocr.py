"""Get the text of an invoice: .txt as is, PDF text layer, or Tesseract OCR for images."""
import shutil
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps
from pypdf import PdfReader

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED = IMAGE_TYPES | {".pdf", ".txt"}


def prepare_image(image: Image.Image) -> Image.Image:
    """Grayscale, upscale small scans, stretch contrast: Tesseract likes ~300 dpi black on white."""
    img = ImageOps.grayscale(image)
    if img.width < 1500:
        img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    return ImageOps.autocontrast(img)


def image_to_text(image: Image.Image, lang: str = "eng") -> str:
    if shutil.which(pytesseract.pytesseract.tesseract_cmd) is None:
        raise RuntimeError("Tesseract is not installed (apt-get install tesseract-ocr / brew install tesseract)")
    return pytesseract.image_to_string(prepare_image(image), lang=lang, config="--psm 6")


def load_text(path: str | Path, lang: str = "eng") -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        if len(text.strip()) > 20:
            return text
        raise RuntimeError(f"{path.name} has no text layer; scan it to an image or use an OCR tool for PDFs")
    if suffix in IMAGE_TYPES:
        with Image.open(path) as img:
            return image_to_text(img, lang)
    raise ValueError(f"unsupported file type {suffix}")
