from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def generate_answer(question, context, api_key):
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.1, 
        groq_api_key=api_key
    )
    
    template = """
    ROLE: You are an Expert Insurance Auditor. 
    
    PHASE 1: VALIDATION
    Determine if the CONTEXT is related to Insurance (Policy, Claim, Terms, or Medical Report).
    - If NOT insurance-related: Reply "❌ This document is not an insurance-related file. I can only audit insurance documents."
    - If YES: Proceed to Phase 2.

    PHASE 2: DYNAMIC AUDIT
    Analyze the document and answer the user's question. 
    - Provide a structured "AUDIT REPORT" based on the specific type of insurance (Life, Health, Motor, etc.).
    - Use headers that make sense for that specific document.
    - Cite sources using [Page X].
    - If a specific detail (like a waiting period) is missing, state that it is not mentioned.
    - End with a 'VERDICT' based on the user's query.

    CONTEXT: {context}
    USER QUESTION: {question}

    FINAL RESPONSE:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = prompt | llm
    
    try:
        response = chain.invoke({"context": context, "question": question})
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"
