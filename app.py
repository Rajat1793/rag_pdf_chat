import streamlit as st

st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("AI Chat Assistant")

# Main page selector
st.header("Choose a Chat Mode")
option = st.radio(
    "What would you like to do?",
    ("Chat with PDF", "Chat with Website"),
    index=0,
    horizontal=True
)

if option == "Chat with PDF":
    from pdf_chat_page import pdf_chat_workflow
    pdf_chat_workflow()
elif option == "Chat with Website":
    from website_chat_page import website_chat_workflow
    website_chat_workflow()
