# not a modular code all working in single file
# use it just for checking the working

import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os
import tempfile
import hashlib
import sqlite3

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize embedding model and OpenAI client
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# SQLite DB for caching
DB_PATH = "file_cache.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS file_cache (
        file_hash TEXT PRIMARY KEY,
        file_name TEXT
    )
""")
conn.commit()

def is_file_processed(file_hash):
    cursor.execute("SELECT 1 FROM file_cache WHERE file_hash = ?", (file_hash,))
    return cursor.fetchone() is not None

def mark_file_as_processed(file_hash, file_name):
    cursor.execute("INSERT OR IGNORE INTO file_cache (file_hash, file_name) VALUES (?, ?)", (file_hash, file_name))
    conn.commit()

# Layout
st.set_page_config(layout="wide")
st.title("📘 PDF Chat Assistant")

# Sidebar for upload and processing
with st.sidebar:
    st.header("📂 Upload & Process")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    process_button = st.button("📄 Process PDF", disabled=not uploaded_file)
    chat_button = st.button("💬 Chat with Me", disabled=True)

# Session state
if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = False
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process PDF
if uploaded_file and process_button:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if is_file_processed(file_hash):
        st.sidebar.info("🔁 File already processed.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = Path(tmp_file.name)

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        QdrantVectorStore.from_documents(
            documents=split_docs,
            url="http://localhost:6333",
            collection_name="learning_ vectors",
            embedding=embedding_model,
            force_recreate=True
        )

        mark_file_as_processed(file_hash, uploaded_file.name)
        st.sidebar.success("✅ PDF processed and indexed!")

    st.session_state.vector_db = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="learning_ vectors",
        embedding=embedding_model,
    )
    st.session_state.chat_ready = True

# Enable chat button after processing
if st.session_state.chat_ready:
    st.sidebar.button("💬 Chat with Me", key="chat_enabled", disabled=False)

# Chat interface
if st.session_state.chat_ready:
    st.subheader("💬 Ask a question about your PDF")
    user_query = st.text_input("You:", key="user_input")

    if user_query:
        search_results = st.session_state.vector_db.similarity_search(query=user_query)

        context = "\n\n\n".join([
            f"Page content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}\nFile Location: {result.metadata.get('source', 'N/A')}"
            for result in search_results
        ])

        SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user queries based on the available context
        retrieved from PDF files along with page contents and page number.

        You should only answer the user based on the following context and navigate the user
        to open the right page number to know more.

        Context:
        {context}
        """

        response = client.chat.completions.create(
            model='gemini-2.0-flash',
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_query}
            ]
        )

        answer = response.choices[0].message.content
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("assistant", answer))

    # Display chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg}")
