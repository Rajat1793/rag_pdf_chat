import streamlit as st
import hashlib
import os
from pdf_processing import process_pdf
from chat_handler import handle_chat
from ui_components import layout_sidebar
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

def pdf_chat_workflow():
    st.subheader("PDF Chat Assistant")
    # Session state initialization
    if "chat_ready" not in st.session_state:
        st.session_state.chat_ready = False
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False

    uploaded_file, process_button, chat_button = layout_sidebar(chat_ready=st.session_state.chat_ready)

    def get_file_hash(file_bytes):
        return hashlib.md5(file_bytes).hexdigest()

    def collection_exists(qdrant_url, qdrant_api_key, collection_name):
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collections = client.get_collections().collections
        return any(col.name == collection_name for col in collections)

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if uploaded_file and process_button:
        file_bytes = uploaded_file.read()
        file_hash = get_file_hash(file_bytes)
        collection_name = f"pdf_{file_hash}"

        if collection_exists(QDRANT_URL, QDRANT_API_KEY, collection_name):
            st.info("🔁 File already processed. Using existing embeddings.")
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                api_key=GOOGLE_API_KEY
            )
            try:
                vector_db = QdrantVectorStore.from_existing_collection(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    collection_name=collection_name,
                    embedding=embedding_model,
                )
                if vector_db is not None:
                    st.session_state.vector_db = vector_db
                    st.session_state.chat_ready = True
                    st.success("Vector store loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load vector store: No collection found.")
            except Exception as e:
                st.error(f"Error loading vector store: {e}")  
        else:
            uploaded_file.seek(0)
            with st.spinner("Processing PDF..."):
                vector_db = process_pdf(uploaded_file, collection_name)
                if vector_db is not None:
                    st.success("PDF processed and indexed!")
                    st.session_state.vector_db = vector_db
                    st.session_state.chat_ready = True
                    st.rerun()

    if st.session_state.chat_ready:
        if chat_button:
            st.session_state.show_chat = True

    if st.session_state.show_chat:
        handle_chat()
