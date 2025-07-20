from services.chat_graph import build_chat_graph
from db.queries import add_chat_message, get_chat

def run_chat(chat_id: int, user_question: str) -> dict:
    chat = get_chat(chat_id)
    file_hashes = [f.hash for f in chat.files]

    graph = build_chat_graph()

    result = graph.invoke({
        "chat_id": chat_id,
        "question": user_question,
        "file_hashes": file_hashes,
    })

    add_chat_message(chat_id, "user", user_question)
    add_chat_message(chat_id, "assistant", result["answer"])

    formatted_images = []
    for i, img in enumerate(result["image_docs"], start=1):
        formatted_images.append({
            "index": i,
            "page": img.metadata.get("page_number"),
            "path": img.metadata.get("image_path"),
            "description": img.page_content.strip()
        })

    return {
        "answer": result["answer"],
        "images": formatted_images
    }
