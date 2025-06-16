from indexing.retriever import retrieve_relevant_chunks
from llm.generator import generate_answer
from db.queries import add_chat_message, get_chat
from db.database import SessionLocal
from db.models import File

def run_chat(chat_id: int, user_question: str) -> tuple[str, list]:
    chat = get_chat(chat_id)
    file_hashes = [f.hash for f in chat.files]

    all_docs = []
    for file_hash in file_hashes:
        docs = retrieve_relevant_chunks(file_hash, user_question)
        all_docs.extend(docs)

    if not all_docs:
        answer = "Nenhuma informação relevante encontrada nos documentos."
    else:
        answer = generate_answer(user_question, all_docs)

    add_chat_message(chat_id, "user", user_question)
    add_chat_message(chat_id, "assistant", answer)

    return answer, all_docs