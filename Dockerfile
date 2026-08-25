FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/
COPY models/emotion_model_v2.h5 ./models/emotion_model_v2.h5

EXPOSE 10000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}