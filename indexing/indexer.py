from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from db.database import SessionLocal
from db.models import Page, File, Image
from pathlib import Path
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_model = HuggingFaceEmbeddings(model_name="local_models/sentence-transformers/all-MiniLM-L6-v2")

MAX_PAGE_CHARS = 1800     # ~1200 tokens, used only with TEXT_CHUNKING=page
MAX_OCR_CHARS = 600       # ~400 tokens

# "subpage" (default): whole page text split into overlapping windows cut at whitespace.
# "page": one chunk per page holding its first MAX_PAGE_CHARS characters (original behavior).
TEXT_CHUNKING = os.getenv("TEXT_CHUNKING", "subpage")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def split_page(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into windows of about `size` characters with `overlap` characters shared
    between neighbors. Cuts fall on whitespace when possible. Returns (start, end, chunk) tuples."""
    spans = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > 0 else end
        spans.append((start, end, text[start:end].strip()))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        sp = text.find(" ", start)
        start = sp + 1 if 0 <= sp < end else start
    return spans

def index_file(file_hash: str):
    db = SessionLocal()
    file = db.query(File).filter(File.hash == file_hash).first()
    pages = db.query(Page).filter(Page.file_hash == file_hash).all()
    images = db.query(Image).filter(Image.file_hash == file_hash).all()
    db.close()

    if not file:
        logger.info(f"File {file_hash} not found.")
        return

    docs = []

    for page in pages:
        text = (page.text or "").strip()
        if not text:
            continue

        if TEXT_CHUNKING == "page":
            spans = [(0, min(len(text), MAX_PAGE_CHARS), text[:MAX_PAGE_CHARS])]
        else:
            spans = split_page(text)

        for start, end, chunk in spans:
            metadata = {
                "file_hash": file_hash,
                "filename": file.filename,
                "page_number": page.page_number,
                "char_start": start,
                "char_end": end,
                "char_count": len(chunk),
                "source": "text"
            }

            docs.append(Document(page_content=chunk, metadata=metadata))

    for image in images:
        parts = []

        if image.caption:
            parts.append(f"Legenda: {image.caption[:300]}")

        if image.ocr_text:
            parts.append(f"OCR: {image.ocr_text[:MAX_OCR_CHARS]}")

        if hasattr(image, "llm_description") and image.llm_description:
            parts.append(f"Descrição IA: {image.llm_description[:400]}")

        content = "\n".join(parts).strip()
        if not content:
            continue

        metadata = {
            "file_hash": file_hash,
            "filename": file.filename,
            "page_number": image.page_number,
            "img_doc_index": image.img_doc_index,
            "img_page_index": image.img_page_index,
            "image_path": image.path,
            "image_id": image.id,
            "image_caption": image.caption,
            "image_description": image.llm_description,
            "ocr_text": image.ocr_text,
            "source": "image"
        }

        docs.append(Document(page_content=content, metadata=metadata))

    if not docs:
        logger.info(f"No indexable content for file {file_hash}")
        return

    index_path = Path("files") / file_hash / "vectorstore"
    index_path.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(str(index_path))

    logger.info(f"Vector index saved to: {index_path}")
