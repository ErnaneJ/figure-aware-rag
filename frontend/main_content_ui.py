import streamlit as st
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
        st.write("⚠️ Chat não encontrado. Por favor, selecione um chat válido na barra lateral.")
        return

    if not all_files_processed(chat_id):
        st.warning("⏳ Processando arquivos, por favor aguarde...")
        return

    messages = get_chat_messages(chat_id)
    render_chat_history(messages)

    if question := st.chat_input("O que você está pensando? 🤔"):
        st.chat_message("user").write(question)

        assistant_msg = st.chat_message("assistant")
        with assistant_msg:
            with st.spinner("🧠 Pensando... (?)"):
                result = run_chat(chat_id, question)
            render_answer_with_images(result["answer"], assistant_msg, streaming=True)
            st.rerun()