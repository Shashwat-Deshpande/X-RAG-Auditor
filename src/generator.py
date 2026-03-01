from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def generate_answer(question, context, api_key):
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0, # Dropped to 0 for maximum strictness
        groq_api_key=api_key
    )
    
    template = """
    ROLE: You are an Expert Insurance Auditor. 
    
    CRITICAL STEP 1: VALIDATION
    Analyze the CONTEXT. Is this an Insurance Policy, Claim, or Medical Report?
    - Indicators of a valid document: Mentions of "Insurer", "Policyholder", "Sum Insured", "Exclusions", "Waiting Periods", or "UIN" codes.
    - If the document is clearly a school assignment, menu, or non-insurance text: 
      REPLY ONLY: "❌ INVALID DOCUMENT: This document is not an insurance policy. Please upload a valid insurance file."
      STOP ALL ANALYSIS.
    
    - If it IS insurance-related: Proceed to Phase 2.

    PHASE 2: DYNAMIC AUDIT
    Provide a professional Audit Report for the user's question:
    1. CATEGORY: Identify if the surgery/condition is "Specified" or "Pre-existing".
    2. TIMELINE: Check the waiting period (e.g., 12/24/36/48 months).
    3. CITATION: Use [Page X] for every rule.
    4. VERDICT: State APPROVED or REJECTED based on the 14-month timeframe vs the policy rule.

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
