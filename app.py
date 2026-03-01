import streamlit as st
import os
from dotenv import load_dotenv
from src.ingestion import process_pdf
from src.retrieval import setup_retrieval
from src.generator import generate_answer

load_dotenv()
# On Streamlit Cloud, this pulls from your 'Secrets'
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="X-RAG Auditor", layout="wide")
st.title("🛡️ X-RAG: Insurance Auditor")

# Sidebar
with st.sidebar:
    st.header("Upload Policy")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        # 1. Ensure the 'data' directory exists on the server
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # 2. Use the ACTUAL filename from the upload (Dynamic!)
        file_path = os.path.join(data_dir, uploaded_file.name)
        
        # 3. Save the file to the server's disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 4. Processing logic
        if "retriever" not in st.session_state or st.button("Process Document"):
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                chunks = process_pdf(file_path)
                st.session_state.retriever = setup_retrieval(chunks)
                st.success(f"✅ {uploaded_file.name} is Ready!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about your policy..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if "retriever" in st.session_state:
        with st.chat_message("assistant"):
            # Retrieve relevant context
            docs = st.session_state.retriever.invoke(user_query)
            
            # Format context with source page numbers
            formatted_context = ""
            for doc in docs:
                page_num = doc.metadata.get("page", 0) + 1
                formatted_context += f"\n[Page {page_num}]: {doc.page_content}\n"
            
            # Generate answer via Groq
            answer = generate_answer(user_query, formatted_context, GROQ_API_KEY)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Please upload and process a document first!")
