# Веб-бэкенд + ядро ассистента
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py document_registry.yaml typical_answers.csv ./
COPY backend/ ./backend/
COPY index/ ./index/

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
