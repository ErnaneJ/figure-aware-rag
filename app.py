import streamlit as st
from db.queries import get_all_chats, get_chat_messages, create_chat
from frontend.chat_ui import render_chat_history
from services.chat_engine import run_chat

st.set_page_config(page_title="Papyrus Chat", layout="wide")
st.title("📚 Papyrus AI")

st.sidebar.title("💬 Seus Chats")

# 🧠 Inicializa o estado
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

# 📄 Listar chats
chats = get_all_chats()
chat_names = [f"{c.id} - {c.name}" for c in chats]
chat_map = {f"{c.id} - {c.name}": c.id for c in chats}

# 🧭 Selecionar chat
selected_chat_name = st.sidebar.selectbox("Escolha um chat", chat_names, index=0 if chats else None)
if selected_chat_name:
    st.session_state.chat_id = chat_map[selected_chat_name]

# ➕ Criar novo chat
with st.sidebar:
    st.markdown("---")
    with st.expander("➕ Novo Chat"):
        filename = st.file_uploader("Suba um PDF", type="pdf", key="upload_pdf")
        if filename and st.button("Criar Chat"):
            file_bytes = filename.read()
            from indexing.file_utils import save_file_if_new
            file_info = save_file_if_new(file_bytes, filename.name)
            new_chat_id = create_chat(file_info["filename"], file_info["hash"])
            st.session_state.chat_id = new_chat_id
            st.rerun()

# 💬 Exibir histórico do chat atual
if st.session_state.chat_id:
    messages = get_chat_messages(st.session_state.chat_id)
    render_chat_history(messages)

    if question := st.chat_input("Pergunte algo sobre o PDF..."):
        with st.spinner("Consultando modelo..."):
            answer, docs = run_chat(st.session_state.chat_id, question)

        st.chat_message("user").markdown(question)
        st.chat_message("assistant").markdown(answer)
    with st.expander("📎 Adicionar novo arquivo a este chat"):
        new_file = st.file_uploader("Selecione um PDF", type="pdf", key="extra_upload")
        if new_file and st.button("Enviar PDF para este chat"):
            file_bytes = new_file.read()
            from indexing.file_utils import save_file_if_new
            from db.queries import associate_file_to_chat

            file_info = save_file_if_new(file_bytes, new_file.name)
            associate_file_to_chat(st.session_state.chat_id, file_info["hash"])
            st.success("Arquivo adicionado com sucesso ao chat!")
else:
    st.info("Crie ou selecione um chat para começar.")
