from sqlalchemy.orm import joinedload
from db.database import SessionLocal
from db.models import Chat, ChatMessage, File
from pathlib import Path

def associate_file_to_chat(chat_id: int, file_hash: str):
    db = SessionLocal()
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    file = db.query(File).filter(File.hash == file_hash).first()
    if chat and file and file not in chat.files:
        chat.files.append(file)
        db.commit()
    db.close()
    
def create_chat(name: str, file_hash: str) -> int:
    db = SessionLocal()
    chat = Chat(name=name)
    file = db.query(File).filter(File.hash == file_hash).first()
    if file:
        chat.files.append(file)
    db.add(chat)
    db.commit()
    chat_id = chat.id
    db.close()
    return chat_id

def get_all_chats():
    db = SessionLocal()
    chats = db.query(Chat).all()
    db.close()
    return chats

def get_chat(chat_id: int):
    db = SessionLocal()
    chat = db.query(Chat).options(joinedload(Chat.files)).filter(Chat.id == chat_id).first()
    db.close()
    return chat

def get_chat_messages(chat_id: int):
    db = SessionLocal()
    msgs = db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id).order_by(ChatMessage.timestamp).all()
    db.close()
    return msgs

def delete_chat_messages(chat_id: int):
    db = SessionLocal()
    db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id).delete()
    db.commit()
    db.close()
    
def add_chat_message(chat_id: int, role: str, content: str, think: str = "", llm_model: str = "", llm_provider: str = ""):
    db = SessionLocal()
    msg = ChatMessage(
        chat_id=chat_id,
        role=role,
        content=content,
        think=think if role == "assistant" else "",
        llm_model=llm_model if role == "assistant" else "",
        llm_provider=llm_provider if role == "assistant" else ""
    )
    db.add(msg)
    db.commit()
    db.close()

    return msg

def get_file_by_hash(file_hash: str):
    db = SessionLocal()
    file = db.query(File).filter(File.hash == file_hash).first()

    if file:
        file_path = Path(file.path)
        if not file_path.exists():
            db.delete(file)
            db.commit()
            db.close()
            return None

    db.close()
    return file

def add_file(file_data: dict):
    db = SessionLocal()
    file = File(**file_data)
    db.add(file)
    db.commit()
    db.close()

def rename_chat(chat_id: int, new_name: str):
    db = SessionLocal()
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat:
        chat.name = new_name
        db.commit()
    db.close()

def delete_chat(chat_id: int):
    db = SessionLocal()
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat:
        db.delete(chat)
        db.commit()
    db.close()

def get_files_for_chat(chat_id: int):
    db = SessionLocal()
    chat = db.query(Chat).options(joinedload(Chat.files)).filter(Chat.id == chat_id).first()
    files = chat.files if chat else []
    db.close()
    return files

def all_files_processed(chat_id: int) -> bool:
    db = SessionLocal()
    chat = db.query(Chat).options(joinedload(Chat.files)).filter(Chat.id == chat_id).first()
    result = all(f.processed for f in chat.files) if chat else False
    db.close()
    return result

def remove_file_from_chat(chat_id: int, file_hash: str):
    db = SessionLocal()
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    file = db.query(File).filter(File.hash == file_hash).first()

    if chat and file and file in chat.files:
        if len(chat.files) <= 1:
            db.close()
            raise ValueError("Um chat precisa ter ao menos um arquivo vinculado.")
        
        chat.files.remove(file)
        db.commit()

    db.close()