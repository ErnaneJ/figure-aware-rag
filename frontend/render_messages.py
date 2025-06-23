import re
import streamlit as st
from db.database import SessionLocal
from db.models import Image

def render_answer_with_images(answer: str, container=None):
    db = SessionLocal()
    cursor = 0
    for match in re.finditer(r"\{\{IMAGE:(\d+)\}\}", answer):
        start, end = match.span()

        # Renderiza texto antes da imagem
        before = answer[cursor:start].strip()
        if before:
            container.markdown(before)

        # Renderiza imagem
        image_id = int(match.group(1))
        image = db.query(Image).filter(Image.id == image_id).first()
        if image:
            description = image.caption or image.llm_description or "Imagem relacionada"
            container.image(image.path, caption=description, width=600)
        else:
            container.markdown(f"❌ [Imagem {image_id} não encontrada]")

        cursor = end

    after = answer[cursor:].strip()
    if after:
        container.markdown(after)

    db.close()

def render_chat_history(messages):
    for m in messages:
        container = st.chat_message(m.role)
        if m.role == "assistant":
            render_answer_with_images(m.content, container)
        else:
            container.markdown(m.content)
