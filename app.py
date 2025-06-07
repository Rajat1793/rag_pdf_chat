import streamlit as st
from pdf_processing import process_pdf
from chat_handler import handle_chat
from ui_components import layout_sidebar
import hashlib
from langchain_google_genai import GoogleGenerativeAIEmbeddings

st.set_page_config(layout="wide")
st.title("PDF Chat Assistant")

uploaded_file, process_button, chat_button = layout_sidebar()

if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = False
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

collection_name = "pdf_chat_collection"

if uploaded_file and process_button:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    with st.spinner("Processing PDF..."):
        vector_db = process_pdf(file_bytes, file_hash, uploaded_file.name, collection_name)
        if vector_db is not None:
            st.sidebar.success("PDF processed and indexed!")
            st.session_state.vector_db = vector_db
            st.session_state.chat_ready = True

if st.session_state.chat_ready:
    st.sidebar.button("Chat with Me", key="chat_enabled", disabled=False)

if st.session_state.chat_ready:
    handle_chat()