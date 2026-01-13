import streamlit as st
from frontend.sidebar_ui import render_sidebar
from frontend.main_content_ui import render_main_content

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

st.set_page_config(page_title="Papyrus Chat")

if chat_id := st.query_params.get("chat_id", [None])[0]:
    st.session_state.chat_id = int(chat_id)

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

render_sidebar()
render_main_content()