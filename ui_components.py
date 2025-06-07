import streamlit as st

def layout_sidebar():
    st.header("Upload & Process")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    process_button = st.button("Process PDF", disabled=not uploaded_file)
    chat_button = st.button("Chat with Me", disabled=True)
    return uploaded_file, process_button, chat_button
