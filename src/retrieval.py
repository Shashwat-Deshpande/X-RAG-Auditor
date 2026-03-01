import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def setup_retrieval(chunks):
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}, # Ensures compatibility with Streamlit Cloud
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # MMR is the gold standard for dense documents
    # We increase k to 10 to give the 70B model more to work with
    # fetch_k=30 gives MMR a larger pool to pick diverse chunks from
    return vector_db.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 10, 
            "fetch_k": 30, 
            "lambda_mult": 0.5 # 0.5 is the sweet spot for diversity vs relevance
        }
    )
