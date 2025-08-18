from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from db.database import SessionLocal
from db.models import Page, File, Image
from pathlib import Path

embedding_model = HuggingFaceEmbeddings(model_name="local_models/sentence-transformers/all-MiniLM-L6-v2")

def index_file(file_hash: str):
    db = SessionLocal()
    file = db.query(File).filter(File.hash == file_hash).first()
    pages = db.query(Page).filter(Page.file_hash == file_hash).all()
    images = db.query(Image).filter(Image.file_hash == file_hash).all()
    db.close()

    if not file:
        print(f"❌ Arquivo {file_hash} não encontrado.")
        return

    docs = []

    for page in pages:
        metadata = {
            "file_hash": file_hash,
            "filename": file.filename,
            "page_number": page.page_number,
            "source": "text"
        }
        docs.append(Document(page_content=page.text, metadata=metadata))

    for image in images:
        parts = []
        if image.caption:
            parts.append(f"Legenda: {image.caption}")
        if image.ocr_text:
            parts.append(f"OCR: {image.ocr_text}")
        if hasattr(image, "llm_description") and image.llm_description:
            parts.append(f"Descrição IA: {image.llm_description}")

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

        # page = [page for page in pages if page.page_number == image.page_number][0]
        # content += f"\n\nConteúdo da página {page.page_number}:\n{page.text}"
        
        docs.append(Document(page_content=content, metadata=metadata))

    if not docs:
        print(f"⚠️ Nenhum conteúdo indexável encontrado para o arquivo {file_hash}")
        return

    index_path = Path("files") / file_hash / "vectorstore"
    index_path.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(str(index_path))

    print(f"✅ Índice vetorial unificado salvo em: {index_path}")
