import streamlit as st

def layout_sidebar(chat_ready=False):
    st.header("Upload & Process")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    process_button = st.button("Process PDF", disabled=not uploaded_file)
    chat_button = st.button("Chat with Me", disabled=not chat_ready)
    return uploaded_file, process_button, chat_button
