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

COPY . .

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]