import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import streamlit as st

api_key = os.getenv("GEMINI_API_KEY")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",api_key=api_key)

def process_pdf(uploaded_file, collection_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    # Limit number of chunks to avoid timeout
    MAX_CHUNKS = 20
    if len(split_docs) > MAX_CHUNKS:
        st.warning(f"PDF is large; only processing the first {MAX_CHUNKS} chunks to avoid timeout.")
        split_docs = split_docs[:MAX_CHUNKS]

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    try:
        vector_db = QdrantVectorStore.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            force_recreate=False
        )
        st.success(f"File indexed and added to vector store: {collection_name}")
    except Exception as e:
        st.error(f"Error during embedding or Qdrant upload: {e}")
        return None

    return vector_db