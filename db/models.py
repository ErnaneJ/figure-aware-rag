from sqlalchemy import Column, String, Integer, ForeignKey, Text, DateTime, Table, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base

class File(Base):
    __tablename__ = "files"

    hash = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    processed = Column(Boolean, default=False)

    pages = relationship("Page", back_populates="file")
    images = relationship("Image", back_populates="file")


class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True)
    file_hash = Column(String, ForeignKey("files.hash"))
    page_number = Column(Integer, nullable=False)
    text = Column(String)

    file = relationship("File", back_populates="pages")

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    file_hash = Column(String, ForeignKey("files.hash"))
    page_number = Column(Integer, nullable=False)
    img_doc_index = Column(Integer, nullable=False)
    img_page_index = Column(Integer, nullable=False)
    ext = Column(String)
    path = Column(String)
    caption = Column(Text, nullable=True)
    ocr_text = Column(Text, nullable=True)
    llm_description = Column(Text, nullable=True)

    file = relationship("File", back_populates="images")

chat_file_association = Table(
    "chat_files", Base.metadata,
    Column("chat_id", Integer, ForeignKey("chats.id")),
    Column("file_hash", String, ForeignKey("files.hash"))
)

class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="chat", cascade="all, delete")
    files = relationship("File", secondary=chat_file_association)

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    role = Column(String)  # "user" ou "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    think = Column(Text)
    llm_model = Column(String)
    llm_provider = Column(String)

    chat = relationship("Chat", back_populates="messages")