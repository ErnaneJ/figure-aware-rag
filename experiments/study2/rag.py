"""Offline replica of the Papyrus AI indexing and retrieval path (indexing/indexer.py,
indexing/retriever.py, services/chat_graph.py, llm/generator.py) with configurable chunking
and image-record composition, used for the unified-settings and ablation runs."""
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
FILENAME = "gsgpy_70_5_3SO.pdf"
MAX_PAGE_CHARS, MAX_OCR_CHARS = 1800, 600
_model = None

def embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _model

def records(name="records.json"):
    return json.loads((HERE / name).read_text())

def text_chunks(pages, mode, size=800, overlap=200):
    """mode='page': first 1800 chars of each page (original). mode='subpage': full page text split
    into overlapping windows of `size` characters, cut at whitespace."""
    docs = []
    for p in pages:
        text = p["text"].strip()
        if not text:
            continue
        if mode == "page":
            docs.append({"text": text[:MAX_PAGE_CHARS], "page": p["page"], "source": "text", "start": 0, "end": min(len(text), MAX_PAGE_CHARS)})
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):
                cut = text.rfind(" ", start + size // 2, end)
                end = cut if cut > 0 else end
            docs.append({"text": text[start:end].strip(), "page": p["page"], "source": "text", "start": start, "end": end})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
            sp = text.find(" ", start)
            start = sp + 1 if 0 <= sp < end else start
    return docs

def image_chunks(images, parts=("caption", "ocr", "desc")):
    docs = []
    for im in images:
        out = []
        if "caption" in parts and im.get("caption"):
            out.append(f"Legenda: {im['caption'][:300]}")
        if "ocr" in parts and im.get("ocr_text"):
            out.append(f"OCR: {im['ocr_text'][:MAX_OCR_CHARS]}")
        if "desc" in parts and im.get("llm_description"):
            out.append(f"Descrição IA: {im['llm_description'][:400]}")
        content = "\n".join(out).strip()
        if content:
            docs.append({"text": content, "page": im["page"], "source": "image", "image_id": im["img_doc_index"]})
    return docs

class Index:
    def __init__(self, docs):
        self.docs = docs
        emb = embedder().encode([d["text"] for d in docs], convert_to_numpy=True).astype("float32")
        # Exact squared-L2 search, equivalent to faiss.IndexFlatL2 used by LangChain's FAISS store.
        self.emb = emb

    def retrieve(self, question, k=5, cutoff=1.8):
        q = embedder().encode([question], convert_to_numpy=True).astype("float32")
        d2 = ((self.emb - q) ** 2).sum(axis=1)
        ids = np.argsort(d2, kind="stable")[:k]
        text_docs, image_docs, seen = [], [], set()
        for d, i in zip(d2[ids], ids):
            if d > cutoff:
                continue
            doc = self.docs[i]
            (image_docs if doc["source"] == "image" else text_docs).append(doc)
        out = []
        for doc in text_docs[:5] + image_docs[:5]:
            key = (doc["page"], doc["text"][:50])
            if key not in seen:
                seen.add(key)
                out.append(doc)
        return out

def build_prompt(question, docs):
    """Verbatim copy of llm/generator.py build_prompt."""
    text_documents = [d for d in docs if d["source"] == "text"]
    image_documents = [d for d in docs if d["source"] == "image"]
    available_files = f"- {FILENAME}" if docs else ""
    text_context = "\n\n".join(f"({FILENAME} – pág. {d['page']})\n{d['text']}" for d in text_documents)
    image_context = "\n\n".join(
        f"(Imagem {d['image_id']} – pág. {d['page']} – {FILENAME})\n{d['text']}" for d in image_documents)
    return f"""You must answer exclusively based on the texts and descriptions below.
        FILES:
        {available_files}

        TEXTS:
        {text_context if text_context else "[none]"}

        IMAGES:
        {image_context if image_context else "[none]"}

        RULES:
        - Use English to answer, even if the question is in another language.
        - For images, write: "Figure X (page Y) of file Z shows:"
        - On the next line, insert exactly:
        {{IMAGE:<id>}}
        - Do not invent information.
        - Always reference the source of the information with: 'Source: file — page'. If there is more than one, list them all. The same applies to images.

        QUESTION:
        {question}
        """

def full_text_context(pages):
    """Full document text, as in experiments/openRouter.py (baseline)."""
    return "".join(f"Page {p['page']}: {p['text'].strip()}\n" for p in pages)
