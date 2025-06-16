from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from db.database import SessionLocal
from db.models import Page, File
from pathlib import Path

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def index_file_pages(file_hash: str):
    db = SessionLocal()
    pages = db.query(Page).filter(Page.file_hash == file_hash).all()
    file = db.query(File).filter(File.hash == file_hash).first()
    db.close()
    if not pages:
        print(f"⚠️ Nenhuma página encontrada para o arquivo {file_hash}")
        return

    docs = []
    for page in pages:
        metadata = {
            "file_hash": file_hash,
            "filename": file.filename,
            "page_number": page.page_number,
        }
        docs.append(Document(page_content=page.text, metadata=metadata) )

    index_path = Path("files") / file_hash / "vectorstore"
    index_path.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(str(index_path))

    print(f"✅ Índice vetorial salvo em: {index_path}")