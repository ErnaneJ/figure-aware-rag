from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

embedding_model = HuggingFaceEmbeddings(model_name="local_models/sentence-transformers/all-MiniLM-L6-v2")

def load_vectorstore(file_hash: str):
    index_path = Path("files") / file_hash / "vectorstore"
    if not index_path.exists():
        raise FileNotFoundError(f"Índice vetorial não encontrado para {file_hash}")
    
    return FAISS.load_local(str(index_path), embedding_model, allow_dangerous_deserialization=True)

def retrieve_relevant_chunks(file_hash: str, question: str, k: int = 5):
    vs = load_vectorstore(file_hash)
    results = vs.similarity_search(question, k=k)

    return results  # Cada item é um langchain_core.documents.documents.Document
