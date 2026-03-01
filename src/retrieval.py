import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def setup_retrieval(chunks):
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={'normalize_embeddings': True}
    )
    
    index_path = "faiss_index"
    
    if os.path.exists(index_path):
        # Load from disk (Instant)
        vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Build and Save to disk
        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(index_path)
    
    # k=12 is good, but let's use MMR to ensure we get different parts of the policy
    return vector_db.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 20}
    )