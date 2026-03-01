import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def setup_retrieval(chunks):
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    index_path = "faiss_index"
    
    # Correcting the scope of vector_db
    if os.path.exists(index_path):
        vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(index_path)
    
    # Now vector_db is guaranteed to be defined
    return vector_db.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.5}
    )
