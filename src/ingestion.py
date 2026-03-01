from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(path):
    loader = PyPDFLoader(path)
    data = loader.load() # This contains page numbers in metadata
    
    # Using 1000/200 split to keep enough context for policy clauses
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
    chunks = text_splitter.split_documents(data)
    
    print(f"✅ Document split into {len(chunks)} chunks with PageIndex preserved.")
    return chunks