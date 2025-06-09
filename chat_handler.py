import streamlit as st
from openai import OpenAI
import os

api_key = os.getenv("GEMINI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

def handle_chat():
    st.subheader("Ask a question about your PDF")
    # Use session state for input
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    user_query = st.text_input("You:", key="user_input")

    if user_query:
        search_results = st.session_state.vector_db.similarity_search(query=user_query)
        context = "\n\n\n".join([
            f"Page content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}\nFile Location: {result.metadata.get('source', 'N/A')}"
            for result in search_results
        ])
        
        SYSTEM_PROMPT = f'''
        You are a helpful AI Assistant who answers user queries based on the available context
        retrieved from PDF files along with page contents and page number.

        You should only answer the user based on the following context and navigate the user
        to open the right page number to know more.

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
        with open("chat_history.txt", "w") as f:
            for role, msg in st.session_state.chat_history:
                f.write(f"{role.capitalize()}: {msg}\n")
        st.download_button("Download Chat History", "chat_history.txt")
