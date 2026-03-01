import streamlit as st
import os
from dotenv import load_dotenv
from src.ingestion import process_pdf
from src.retrieval import setup_retrieval
from src.generator import generate_answer

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="X-RAG Auditor", layout="wide")
st.title("🛡️ X-RAG: Insurance Auditor")

# Sidebar
with st.sidebar:
    st.header("Upload Policy")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        file_path = os.path.join("data", "policy.pdf")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # We only process if explicitly clicked or if retriever isn't in memory
        if st.button("Process Document") or "retriever" not in st.session_state:
            with st.spinner("Analyzing PDF..."):
                chunks = process_pdf(file_path)
                st.session_state.retriever = setup_retrieval(chunks)
                st.success("✅ Knowledge Base Ready!")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Query..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if "retriever" in st.session_state:
        with st.chat_message("assistant"):
            docs = st.session_state.retriever.invoke(user_query)
            
            # FORMATTING CONTEXT WITH PAGE INDEX
            formatted_context = ""
            for doc in docs:
                # Adding 1 because PDF indexing starts at 0
                page_num = doc.metadata.get("page", 0) + 1
                formatted_context += f"\n[Page {page_num}]: {doc.page_content}\n"
            
            answer = generate_answer(user_query, formatted_context, GROQ_API_KEY)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Upload a document first!")