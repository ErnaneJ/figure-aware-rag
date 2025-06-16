import streamlit as st

def render_chat_history(messages):
    for m in messages:
        with st.chat_message(m.role):
            st.markdown(m.content)
