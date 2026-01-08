import streamlit as st
from db.queries import (
    get_all_chats, create_chat, delete_chat, rename_chat, all_files_processed,
    get_files_for_chat, remove_file_from_chat, associate_file_to_chat, get_file_by_hash, delete_chat_messages
)
from indexing.file_utils import save_file_if_new
import pandas as pd
from streamlit_pdf_viewer import pdf_viewer

def render_sidebar():
    st.sidebar.title("📚 Papyrus AI")
    st.sidebar.write("Converse com seus PDFs de forma inteligente!")

    # ➕ Criar novo chat
    with st.sidebar.expander("➕ Criar novo chat"):
        new_chat_name = st.text_input("Título do chat", key="new_chat_name")
        new_pdfs = st.file_uploader("Arquivo(s) PDF", type="pdf", key="upload_new_chat", accept_multiple_files=True)
        if new_chat_name and new_pdfs and st.button("Criar", width='stretch'):
            for new_pdf in new_pdfs:
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
           f"{chat.name.title()}" if all_files_processed_for_this_chat else f"{chat.name.title()} ⏳",
           key=f"select_{chat.id}", width='stretch', disabled=not all_files_processed_for_this_chat
        ):
            st.query_params.chat_id = chat.id
            st.session_state.chat_id = chat.id
            st.rerun()

        if cols[1].button("✏️", key=f"edit_btn_{chat.id}"):
            st.session_state.dialog_editing_chat_id = chat.id

        if cols[2].button("🗑️", key=f"delete_btn_{chat.id}"):
            st.session_state.dialog_confirm_delete_id = chat.id

        # Renomear
        if st.session_state.get("dialog_editing_chat_id") == chat.id:
            st.session_state.dialog_viewing_pdf = None
            st.session_state.dialog_confirm_delete_id = None
            edit_chat_dialog(chat)

        # Confirmar exclusão
        if st.session_state.get("dialog_confirm_delete_id") == chat.id:
            st.session_state.dialog_viewing_pdf = None
            st.session_state.dialog_editing_chat_id = None
            delete_chat_dialog(chat)

        if chat_id == chat.id:
          files = get_files_for_chat(chat_id)
          with st.sidebar.expander("📎 Arquivos vinculados", expanded=True):
            files_table = pd.DataFrame({
              "Status": ["✅" if f.processed else "⏳" for f in files],
              "Arquivo": [f.filename for f in files]
            })
            st.dataframe(files_table, width='stretch', hide_index=True)
            
            if st.session_state.get("dialog_viewing_pdf") is not None:
              st.session_state.dialog_editing_chat_id = None
              st.session_state.dialog_confirm_delete_id = None
              view_pdf_dialog(st.session_state.dialog_viewing_pdf)

@st.dialog("🗑️ Apagar Chat")
def delete_chat_dialog(chat):
  st.write(f"Tem certeza que deseja excluir o chat '`{chat.name}`'?")
  col1, col2 = st.columns([1, 1])
  if col1.button("✅ Confirmar", key=f"confirm_delete_{chat.id}", width='stretch') and "dialog_confirm_delete_id" in st.session_state:
    delete_chat(chat.id)
    st.session_state["dialog_confirm_delete_id"] = None
    st.rerun()
  if col2.button("❌ Cancelar", key=f"cancel_editing_chat_{chat.id}", width='stretch') and "dialog_confirm_delete_id" in st.session_state:
    st.session_state["dialog_confirm_delete_id"] = None
    st.rerun()

@st.dialog("✏️ Editar Chat")
def edit_chat_dialog(chat):
    st.write(f"Título do Chat")
    col1, col2 = st.columns([2, 1])
    new_name = col1.text_input("Título", value=chat.name, key=f"new_name_input_{chat.id}", label_visibility="collapsed")
    if col2.button("✅ Atualizar", key=f"editing_chat_{chat.id}", width='stretch'):
      rename_chat(chat.id, new_name)
      st.session_state["dialog_editing_chat_id"] = None
      st.session_state["dialog_viewing_pdf"] = None
      st.rerun()

    if st.button("🧹 Limpar Mensagens do Chat", key=f"delete_messages_{chat.id}", width='stretch'):
        delete_chat_messages(chat.id)
        st.session_state["dialog_editing_chat_id"] = None
        st.rerun()

    st.subheader("Arquivos Já Vinculados")

    # 🗂️ Listar arquivos do chat
    files = get_files_for_chat(chat.id)
    if not files:
      st.info("Nenhum arquivo vinculado a este chat.")
    else:
      for file in files:
        file_row = st.columns([6, 1, 1])
        label = f"`{'✅' if file.processed else '⏳'} {file.filename[-43:]}`"
        
        file_row[0].markdown(label)

        if file_row[1].button("📄", key=f"view_file_{file.hash}_{chat.id}", width='stretch'):
            st.session_state["dialog_editing_chat_id"] = None
            st.session_state["dialog_viewing_pdf"] = file.hash
            st.rerun()

        if file_row[2].button("🗑️", key=f"remove_file_{file.hash}_{chat.id}", disabled=not file.processed):
            try:
                remove_file_from_chat(chat.id, file.hash)
                st.success(f"Removido: {file.filename}")
            except ValueError as e:
                st.warning(str(e))

    st.subheader("Adicionar Novo(s) Arquivo(s)")
    new_files = st.file_uploader("Selecione", type="pdf", key=f"new_file_upload_{chat.id}", accept_multiple_files=True)
    if new_files and st.button("📎 Adicionar Arquivo(s)", key=f"add_file_{chat.id}", width='stretch'):
      for new_file in new_files:
          file_bytes = new_file.read()
          file_info = save_file_if_new(file_bytes, new_file.name)
          associate_file_to_chat(chat.id, file_info["hash"])
      st.success("Arquivo(s) adicionado(s) com sucesso.")
      st.rerun()

    if st.button("❌ Cancelar", key=f"cancel_edit_chat_{chat.id}", width='stretch'):
        st.session_state["dialog_editing_chat_id"] = None
        st.rerun()

@st.dialog("👀 Visualizar Arquivo", width="large")
def view_pdf_dialog(file_hash):
    file = get_file_by_hash(file_hash)
    st.write(f"- 📄 **Arquivo:** `{file.filename}`\n- 🔗 **Hash:** `{file.hash}`\n- ⚙️ **Status:** `{'✅ Processado' if file.processed else '⏳ Processando'}`")

    pdfBytes = open(file.path, "rb").read()
    pdf_viewer(pdfBytes, height=800, zoom_level=1.0, key=f"pdf_viewer_{file.hash}")
    if st.button("❌ Fechar", width='stretch'):
        st.session_state["dialog_viewing_pdf"] = None
        st.rerun()