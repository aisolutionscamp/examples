FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt
COPY app ./app
COPY train.py .
# Mount or copy your fine-tuned models to /srv/models/urgency and /srv/models/department.
ENV URGENCY_MODEL_DIR=/srv/models/urgency DEPARTMENT_MODEL_DIR=/srv/models/department
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
