"""REST API: upload a document, get structured data back.

    uvicorn docproc.api:app --app-dir src --port 8000
"""
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from .pipeline import process_document
from .text import IMAGE_TYPES, OCRUnavailable, ocr_available

MAX_BYTES = 20 * 1024 * 1024
ALLOWED = IMAGE_TYPES | {".pdf"}

app = FastAPI(title="Document processing (Tesseract + spaCy)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ocr": ocr_available()}


@app.post("/documents")
async def process(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"allowed file types: {', '.join(sorted(ALLOWED))}")
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "file larger than 20 MB")
    # random temp name: the client's file name never touches the file system
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            result = process_document(tmp.name)
        except OCRUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:  # unreadable or broken files
            raise HTTPException(422, f"could not read document: {type(exc).__name__}") from exc
    return json.loads(json.dumps({"filename": file.filename, **result}, default=str))
