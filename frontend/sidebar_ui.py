import streamlit as st
from db.queries import (
    get_all_chats, create_chat, delete_chat, rename_chat, all_files_processed,
    get_files_for_chat, remove_file_from_chat, associate_file_to_chat, get_file_by_hash, delete_chat_messages
)
from indexing.file_utils import save_file_if_new
import pandas as pd
from streamlit_pdf_viewer import pdf_viewer

def render_sidebar():
    st.sidebar.title("Papyrus AI")
    st.sidebar.write("Chat with your PDFs.")

    with st.sidebar.expander("New chat"):
        new_chat_name = st.text_input("Chat title", key="new_chat_name")
        new_pdfs = st.file_uploader("PDF file(s)", type="pdf", key="upload_new_chat", accept_multiple_files=True)
        if new_chat_name and new_pdfs and st.button("Create", width='stretch'):
            new_chat_id = create_chat(new_chat_name, "-- NONE --")
            for new_pdf in new_pdfs:
                file_bytes = new_pdf.read()
                file_info = save_file_if_new(file_bytes, new_pdf.name)
                associate_file_to_chat(new_chat_id, file_info["hash"])
            st.query_params.chat_id = new_chat_id
            st.rerun()

    st.sidebar.markdown('---')

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

        if st.session_state.get("dialog_editing_chat_id") == chat.id:
            st.session_state.dialog_viewing_pdf = None
            st.session_state.dialog_confirm_delete_id = None
            edit_chat_dialog(chat)

        if st.session_state.get("dialog_confirm_delete_id") == chat.id:
            st.session_state.dialog_viewing_pdf = None
            st.session_state.dialog_editing_chat_id = None
            delete_chat_dialog(chat)

        if chat_id == chat.id:
          files = get_files_for_chat(chat_id)
          with st.sidebar.expander("Files", expanded=True):
            files_table = pd.DataFrame({
              "Status": ["✅" if f.processed else "⏳" for f in files],
              "File": [f.filename for f in files]
            })
            st.dataframe(files_table, width='stretch', hide_index=True)
            
            if st.session_state.get("dialog_viewing_pdf") is not None:
              st.session_state.dialog_editing_chat_id = None
              st.session_state.dialog_confirm_delete_id = None
              view_pdf_dialog(st.session_state.dialog_viewing_pdf)

@st.dialog("Delete chat")
def delete_chat_dialog(chat):
  st.write(f"Delete the chat '`{chat.name}`'?")
  col1, col2 = st.columns([1, 1])
  if col1.button("Confirm", key=f"confirm_delete_{chat.id}", width='stretch') and "dialog_confirm_delete_id" in st.session_state:
    delete_chat(chat.id)
    st.session_state["dialog_confirm_delete_id"] = None
    st.rerun()
  if col2.button("Cancel", key=f"cancel_editing_chat_{chat.id}", width='stretch') and "dialog_confirm_delete_id" in st.session_state:
    st.session_state["dialog_confirm_delete_id"] = None
    st.rerun()

@st.dialog("Edit chat")
def edit_chat_dialog(chat):
    st.write(f"Chat title")
    col1, col2 = st.columns([2, 1])
    new_name = col1.text_input("Title", value=chat.name, key=f"new_name_input_{chat.id}", label_visibility="collapsed")
    if col2.button("Save", key=f"editing_chat_{chat.id}", width='stretch'):
      rename_chat(chat.id, new_name)
      st.session_state["dialog_editing_chat_id"] = None
      st.session_state["dialog_viewing_pdf"] = None
      st.rerun()

    if st.button("Clear chat messages", key=f"delete_messages_{chat.id}", width='stretch'):
        delete_chat_messages(chat.id)
        st.session_state["dialog_editing_chat_id"] = None
        st.rerun()

    st.subheader("Linked files")

    files = get_files_for_chat(chat.id)
    if not files:
      st.info("No files linked to this chat.")
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
                st.success(f"Removed: {file.filename}")
            except ValueError as e:
                st.warning(str(e))

    st.subheader("Add files")
    new_files = st.file_uploader("Select", type="pdf", key=f"new_file_upload_{chat.id}", accept_multiple_files=True)
    if new_files and st.button("Add file(s)", key=f"add_file_{chat.id}", width='stretch'):
      for new_file in new_files:
          file_bytes = new_file.read()
          file_info = save_file_if_new(file_bytes, new_file.name)
          associate_file_to_chat(chat.id, file_info["hash"])
      st.success("File(s) added.")
      st.rerun()

    if st.button("Cancel", key=f"cancel_edit_chat_{chat.id}", width='stretch'):
        st.session_state["dialog_editing_chat_id"] = None
        st.rerun()

@st.dialog("View file", width="large")
def view_pdf_dialog(file_hash):
    file = get_file_by_hash(file_hash)
    st.write(f"- **File:** `{file.filename}`\n- **Hash:** `{file.hash}`\n- **Status:** `{'Processed' if file.processed else 'Processing'}`")

    pdfBytes = open(file.path, "rb").read()
    pdf_viewer(pdfBytes, height=800, zoom_level=1.0, key=f"pdf_viewer_{file.hash}")
    if st.button("Close", width='stretch'):
        st.session_state["dialog_viewing_pdf"] = None
        st.rerun()