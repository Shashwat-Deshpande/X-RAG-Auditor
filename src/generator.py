from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def generate_answer(question, context, api_key):
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0, # Dropped to 0 for maximum strictness
        groq_api_key=api_key
    )
    
    template = """
    ROLE: You are a Gatekeeper and Insurance Auditor. 
    
    CRITICAL STEP 1: VALIDATION
    Analyze the CONTEXT below. Is this an Insurance Policy, Insurance Claim, or Medical Report?
    - If it is a college assignment, school work, restaurant menu, or any non-insurance text: 
      YOU MUST ONLY REPLY WITH: "❌ INVALID DOCUMENT: This document is not an insurance policy. Please upload a valid insurance file to proceed."
      STOP ALL FURTHER ANALYSIS. DO NOT PROCEED TO PHASE 2.
    
    - If it IS insurance-related: Proceed to Phase 2.

    PHASE 2: DYNAMIC AUDIT (Only run if Phase 1 is 'Insurance')
    Analyze the document and answer the user's question:
    - Provide a structured audit based on the specific insurance type found.
    - Cite sources using [Page X].
    - If the user asks for something not in the document, say it is missing.
    - End with a clear 'VERDICT'.

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
