import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from qdrant_client import QdrantClient
import hashlib

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def get_url_hash(url):
    return hashlib.md5(url.encode()).hexdigest()

def collection_exists(qdrant_url, qdrant_api_key, collection_name):
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    collections = client.get_collections().collections
    return any(col.name == collection_name for col in collections)

def website_chat_workflow():
    st.title("Chat with Website")

    url = st.text_input("Enter website URL to chat with:")
    process_button = st.button("Process Website")
    if "chat_ready" not in st.session_state:
        st.session_state.chat_ready = False
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False

    if url and process_button:
        url_hash = get_url_hash(url)
        collection_name = f"website_{url_hash}"

        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            api_key=GOOGLE_API_KEY
        )

        if collection_exists(QDRANT_URL, QDRANT_API_KEY, collection_name):
            st.success("Website already processed. Using existing embeddings.")
            st.session_state.vector_db = QdrantVectorStore.from_existing_collection(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=collection_name,
                embedding=embedding_model,
            )
            st.session_state.chat_ready = True
            st.session_state.show_chat = False
            st.rerun()
        else:
            with st.spinner("Loading and embedding website content..."):
                loader = WebBaseLoader(url)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                split_docs = text_splitter.split_documents(docs)

                # Limit number of chunks to avoid timeout
                MAX_CHUNKS = 20
                if len(split_docs) > MAX_CHUNKS:
                    st.warning(f"Website is large; only processing the first {MAX_CHUNKS} chunks to avoid timeout.")
                    split_docs = split_docs[:MAX_CHUNKS]

                try:
                    vector_db = QdrantVectorStore.from_documents(
                        documents=split_docs,
                        embedding=embedding_model,
                        url=QDRANT_URL,
                        api_key=QDRANT_API_KEY,
                        collection_name=collection_name,
                        force_recreate=False
                    )
                    st.success("Website processed and indexed!")
                    st.session_state.vector_db = vector_db
                    st.session_state.chat_ready = True
                    st.session_state.show_chat = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during embedding or Qdrant upload: {e}")

    if st.session_state.chat_ready:
        chat_button = st.button("Chat with Website")
        if chat_button:
            st.session_state.show_chat = True

    if st.session_state.show_chat:
        handle_website_chat()

def handle_website_chat():
    client = OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    st.subheader("Ask a question about the website")
    user_query = st.text_input("You:", key="website_user_input")

    if user_query:
        search_results = st.session_state.vector_db.similarity_search(query=user_query)
        context = "\n\n\n".join([
            f"Content: {result.page_content}\nSource: {result.metadata.get('source', 'N/A')}"
            for result in search_results
        ])

        SYSTEM_PROMPT = f'''
        You are a helpful AI Assistant who answers user queries based on the available context
        retrieved from the website.

        Context:
        {context}
        '''

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

    # Display chat history (newest at top)
    for role, msg in reversed(st.session_state.chat_history):
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

    if st.button("Export Chat History"):
        with open("website_chat_history.txt", "w") as f:
            for role, msg in st.session_state.chat_history:
                f.write(f"{role.capitalize()}: {msg}\n")
        st.download_button("Download Chat History", "website_chat_history.txt")
