import streamlit as st
from db.queries import (
    get_all_chats, create_chat, delete_chat, rename_chat, all_files_processed,
    get_files_for_chat, remove_file_from_chat, associate_file_to_chat
)
from indexing.file_utils import save_file_if_new
import pandas as pd

def render_sidebar():
    st.sidebar.title("📚 Papyrus AI")
    st.sidebar.write("Converse com seus PDFs de forma inteligente!")

    # ➕ Criar novo chat
    with st.sidebar.expander("➕ Criar novo chat"):
        new_chat_name = st.text_input("Título do chat", key="new_chat_name")
        new_pdf = st.file_uploader("Arquivo(s) PDF", type="pdf", key="upload_new_chat")
        if new_chat_name and new_pdf and st.button("Criar", use_container_width=True):
            file_bytes = new_pdf.read()
            file_info = save_file_if_new(file_bytes, new_pdf.name)
            new_chat_id = create_chat(new_chat_name, file_info["hash"])
            st.query_params.chat_id = new_chat_id
            st.rerun()

    st.sidebar.markdown('---')

    # 📋 Listagem de chats
    chats = get_all_chats()
    chat_id = st.session_state.chat_id
    for chat in chats:
        row = st.sidebar.container()
        cols = row.columns([6, 1, 1])

        all_files_processed_for_this_chat = all_files_processed(chat.id)
        if cols[0].button(
           chat.name if all_files_processed_for_this_chat else f"{chat.name} ⏳",
           key=f"select_{chat.id}", use_container_width=True, disabled=not all_files_processed_for_this_chat
        ):
            st.query_params.chat_id = chat.id
            st.session_state.chat_id = chat.id
            st.rerun()

        if cols[1].button("✏️", key=f"edit_btn_{chat.id}"):
            st.session_state.editing_chat_id = chat.id

        if cols[2].button("🗑️", key=f"delete_btn_{chat.id}"):
            st.session_state.confirm_delete_id = chat.id

        # Renomear
        if st.session_state.get("editing_chat_id") == chat.id:
            edit_chat_dialog(chat)

        # Confirmar exclusão
        if st.session_state.get("confirm_delete_id") == chat.id:
            delete_chat_dialog(chat)

        if chat_id == chat.id:
          files = get_files_for_chat(chat_id)
          with st.sidebar.expander("📎 Arquivos vinculados", expanded=True):
            files_table = pd.DataFrame({
              "Status": ["✅" if f.processed else "⏳" for f in files],
              "Arquivo": [f.filename for f in files]
            })
            st.dataframe(files_table, use_container_width=True, hide_index=True)

@st.dialog("🗑️ Apagar Chat")
def delete_chat_dialog(chat):
  st.write(f"Tem certeza que deseja excluir o chat '`{chat.name}`'?")
  col1, col2 = st.columns([1, 1])
  if col1.button("✅ Confirmar", key=f"confirm_delete_{chat.id}", use_container_width=True) and "confirm_delete_id" in st.session_state:
    delete_chat(chat.id)
    st.session_state["confirm_delete_id"] = None
    st.rerun()
  if col2.button("❌ Cancelar", key=f"cancel_editing_chat_{chat.id}", use_container_width=True) and "confirm_delete_id" in st.session_state:
    st.session_state["confirm_delete_id"] = None
    st.rerun()

@st.dialog("✏️ Editar Chat")
def edit_chat_dialog(chat):
    st.write(f"Título do Chat")
    col1, col2 = st.columns([2, 1])
    new_name = col1.text_input("Título", value=chat.name, key=f"new_name_input_{chat.id}", label_visibility="collapsed")
    if col2.button("✅ Atualizar", key=f"editing_chat_{chat.id}", use_container_width=True):
      rename_chat(chat.id, new_name)
      st.session_state["editing_chat_id"] = None
      st.rerun()

    st.subheader("Arquivos Já Vinculados")

    # 🗂️ Listar arquivos do chat
    files = get_files_for_chat(chat.id)
    if not files:
      st.info("Nenhum arquivo vinculado a este chat.")
    else:
      for file in files:
        file_row = st.columns([6, 1])
        file_row[0].button(f"`{'✅' if file.processed else '⏳'} {file.filename[-43:]}`", disabled=True, key=f"file_name_{file.hash}_{chat.id}", use_container_width=True)
        if file_row[1].button("🗑️", key=f"remove_file_{file.hash}_{chat.id}", disabled=not file.processed):
          try:
            remove_file_from_chat(chat.id, file.hash)
            st.success(f"Removido: {file.filename}")
          except ValueError as e:
            st.warning(str(e))

    st.subheader("Adicionar Novo(s) Arquivo(s)")
    new_file = st.file_uploader("Selecione um PDF", type="pdf", key=f"new_file_upload_{chat.id}")
    if new_file and st.button("📎 Adicionar Arquivo(s)", key=f"add_file_{chat.id}", use_container_width=True):
      file_bytes = new_file.read()
      file_info = save_file_if_new(file_bytes, new_file.name)
      associate_file_to_chat(chat.id, file_info["hash"])
      st.success("Arquivo adicionado com sucesso.")
      st.rerun()

    if st.button("❌ Cancelar", key=f"cancel_edit_chat_{chat.id}", use_container_width=True):
        st.session_state["editing_chat_id"] = None
        st.rerun()
