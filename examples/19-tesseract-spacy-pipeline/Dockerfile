FROM python:3.12-slim
# tesseract-ocr + English data; add e.g. tesseract-ocr-deu for German documents
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "docproc.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
