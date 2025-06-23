# frontend/main_content.py
import streamlit as st
import time
from db.queries import (
    get_chat, get_chat_messages, all_files_processed
)
from frontend.render_messages import render_chat_history, render_answer_with_images
from services.chat_engine import run_chat

def render_main_content():
    if st.session_state.get("chat_id") is None:
        st.title("📚 Papyrus AI")
        st.write("Converse com seus PDFs de forma inteligente!.")
        st.info("👈🏼 Selecione um chat na barra lateral ou crie um novo.")
        return
    
    chat_id = st.session_state.chat_id
    chat = get_chat(chat_id)
    if not chat:
        st.session_state.chat_id = None
        st.rerun()

    # 🔁 Espera até que todos os arquivos estejam prontos
    with st.spinner("Processando arquivo(s)...", show_time=True):
        while not all_files_processed(chat_id):
            time.sleep(10)
            st.rerun()

    # 💬 Histórico de mensagens
    messages = get_chat_messages(chat_id)
    render_chat_history(messages)

    if question := st.chat_input("O que você está pensando? 🤔"):
        st.chat_message("user").markdown(question)

        assistant_msg = st.chat_message("assistant")
        with assistant_msg:
            with st.spinner("🧠 Pensando... (?)"):
                result = run_chat(chat_id, question)
        render_answer_with_images(result["answer"], assistant_msg)