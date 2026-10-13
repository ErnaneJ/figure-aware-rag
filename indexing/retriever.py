from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_model = HuggingFaceEmbeddings(model_name="local_models/sentence-transformers/all-MiniLM-L6-v2")

def load_vectorstore(file_hash: str):
    index_path = Path("files") / file_hash / "vectorstore"
    if not index_path.exists():
        raise FileNotFoundError(f"Índice vetorial não encontrado para {file_hash}")
    
    return FAISS.load_local(str(index_path), embedding_model, allow_dangerous_deserialization=True)

def retrieve_relevant_chunks(file_hash: str, question: str, k: int = 100):
    vs = load_vectorstore(file_hash)
    results = vs.similarity_search_with_score(question, k=k)

    text_docs = []
    image_docs = []

    for doc, score in results:
        logger.info(
            f"[RETRIEVER] score={score:.4f} | page={doc.metadata.get('page_number')} | preview={doc.page_content[:80]}"
        )
        
        doc.metadata["score"] = score

        if score > 1.8:
            continue

        if doc.metadata.get("source") == "image":
            image_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs[:5], image_docs[:5]