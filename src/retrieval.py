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
    
    # Check if we already have a saved index to save time/resources
    if os.path.exists(index_path):
        vector_db = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        # If no index exists, create it from the provided chunks
        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(index_path)
    
    # Return the retriever with the optimized MMR settings we discussed
    return vector_db.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 10, 
            "fetch_k": 30, 
            "lambda_mult": 0.5
        }
    )
