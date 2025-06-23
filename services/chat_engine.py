from indexing.retriever import retrieve_relevant_chunks
from llm.generator import generate_answer
from db.queries import add_chat_message, get_chat
from db.database import SessionLocal
from db.models import File

def run_chat(chat_id: int, user_question: str) -> dict:
    chat = get_chat(chat_id)
    file_hashes = [f.hash for f in chat.files]

    all_docs = []
    for file_hash in file_hashes:
        docs = retrieve_relevant_chunks(file_hash, user_question)
        all_docs.extend(docs)

    text_docs = [doc for doc in all_docs if doc.metadata.get("source") == "text"]
    image_docs = [doc for doc in all_docs if doc.metadata.get("source") == "image"]

    if not text_docs and not image_docs:
        answer = "Nenhuma informação relevante encontrada nos documentos."
    else:
        answer = generate_answer(user_question, text_docs, image_docs)

    add_chat_message(chat_id, "user", user_question)
    add_chat_message(chat_id, "assistant", answer)

    formatted_images = []
    for i, img in enumerate(image_docs, start=1):
        formatted_images.append({
            "index": i,
            "page": img.metadata.get("page_number"),
            "path": img.metadata.get("image_path"),
            "description": img.page_content.strip()
        })

    return {
        "answer": answer,
        "images": formatted_images
    }
