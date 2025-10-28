FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-por \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    pkg-config \
    python3-dev \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p files local_models

COPY utils/download_all_miniLM.py .
RUN python download_all_miniLM.py

COPY . .

RUN chmod -R 755 /app/files /app/local_models

RUN groupadd -r papyrus && useradd -r -g papyrus papyrus \
    && chown -R papyrus:papyrus /app

USER papyrus

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]