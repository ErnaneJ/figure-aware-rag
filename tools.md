# TOOLS - PapyrusAI

## Libs (Python)

- streamlit
- sqlalchemy
- psycopg2-binary
- python-dotenv
- celery
- redis
- PyMuPDF
- langchain
- faiss-cpu
- sentence-transformers
- langchain-community
- langchain-huggingface
- matplotlib
- langchain-google-genai
- google-generativeai
- pytesseract
- Pillow
- google-genai
- langgraph
- streamlit_pdf_viewer
- langchain_ollama

## Containers (Docker)

### App

- Image: python:3.10-slim

#### w/

- gcc
- build-essential
- tesseract-ocr
- tesseract-ocr-por
- libtesseract-dev
- libleptonica-dev
- poppler-utils
- pkg-config
- python3-dev

### worker

- Image: python:3.10-slim

#### w/

- gcc
- build-essential
- tesseract-ocr
- tesseract-ocr-por
- libtesseract-dev
- libleptonica-dev
- poppler-utils
- pkg-config
- python3-dev

### DB

- Image: postgres:14

### Redis

- Image: redis:6

### Pretendo usar

#### Ollama (experimental)

- Image: ollama

#### Banco Vetorial

(nenhum em mente ainda)
