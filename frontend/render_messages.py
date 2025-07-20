import re
import streamlit as st
from db.database import SessionLocal
from db.models import Image
import time

def stream_response_data(text: str):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

def render_answer_with_images(answer: str, container=None, streaming=False):
    db = SessionLocal()
    cursor = 0
    for match in re.finditer(r"\{\{IMAGE:(\d+)\}\}", answer):
        start, end = match.span()

        # Text
        before = answer[cursor:start].strip()
        if before:
            if streaming:
                container.write_stream(stream_response_data(before))
            else:
                container.write(before)

        # Image
        image_id = int(match.group(1))
        image = db.query(Image).filter(Image.id == image_id).first()
        if image:
            description = image.caption or image.llm_description or "Imagem relacionada"
            container.image(image.path, caption=description, width=600)
        else:
            if streaming:
                container.write_stream(stream_response_data(f"❌ [Imagem {image_id} não encontrada]"))
            else:
                container.write(f"❌ [Imagem {image_id} não encontrada]")

        cursor = end

    # Text
    after = answer[cursor:].strip()
    if after:
        if streaming:
            container.write_stream(stream_response_data(after))
        else:
            container.write(after)

    db.close()

def render_chat_history(messages):
    for m in messages:
        container = st.chat_message(m.role)
        if m.role == "assistant":
            render_answer_with_images(m.content, container, streaming=False)
        else:
            container.write(m.content)
