import os
from dotenv import load_dotenv
from src.ingestion import process_pdf
from src.retrieval import setup_retrieval
from src.generator import generate_answer

# Load environment variables (GROQ_API_KEY)
load_dotenv()

def run_rag_test():
    # 1. Fetch Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in .env file.")
        return

    # 2. Define PDF Path
    pdf_path = "data/policy.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found. Please place your policy PDF in the 'data' folder.")
        return

    # 3. Process and Split PDF
    # The ingestion script now preserves the 'page' metadata
    chunks = process_pdf(pdf_path)
    
    # 4. Setup Retrieval (Persistence Enabled)
    print("🔄 Initializing Vector Database (FAISS)...")
    retriever = setup_retrieval(chunks)
    
    # 5. Define Test Query
    # Testing the specific case of Bariatric surgery to check for Page Citations
    user_query = "Audit the claim: Insured has 25 months tenure, has Obesity, and wants Bariatric Surgery in an AYUSH hospital."
    
    print(f"\n🔍 Testing Auditor Logic for: '{user_query}'...")
    
    # 6. Retrieve Documents and Format with Page Numbers
    # This matches the logic in your app.py
    retrieved_docs = retriever.invoke(user_query)
    
    formatted_context = ""
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", 0) + 1  # Standardizing to 1-based indexing
        formatted_context += f"\n[Source: Page {page_num}]: {doc.page_content}\n"
    
    # 7. Generate final response
    print("🤖 Generating Audit Verdict...")
    final_answer = generate_answer(user_query, formatted_context, api_key)
    
    # 8. Output Result
    print("\n" + "="*50)
    print("🛡️ X-RAG AUDIT VERDICT:")
    print("="*50)
    print(final_answer)
    print("="*50 + "\n")

if __name__ == "__main__":
    run_rag_test()