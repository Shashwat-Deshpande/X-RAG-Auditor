from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def generate_answer(question, context, api_key):
    # Using 70b-specdec for high-precision math and reasoning
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0, 
        groq_api_key=api_key
    )
    
    template = """
    ROLE: You are an Expert Insurance Auditor and Financial Analyst.

    PHASE 1: CONTENT VALIDATION
    - If the uploaded PDF context does not contain terms like "Policy", "Insurer", "Benefit", "Exclusion", or "Sum Insured", 
      you MUST start your response with: "⚠️ NOTICE: This document does not appear to be an insurance policy or insurance-related."
    - Continue to answer the question regardless, but provide this warning first.

    PHASE 2: MATHEMATICAL REASONING
    - If the user asks for a calculation (sums, square roots, percentages), solve it step-by-step.
    - Example: 15 + 27 = 42.
    
    PHASE 3: POLICY AUDIT
    - If the question is about a claim (like Glaucoma/Cataract), search for "Waiting Periods" (Section 3).
    - If a 14-month claim is made against a 24-month rule, the verdict is REJECTED.
    - Cite Page Numbers [Page X].

    CONTEXT: 
    {context}

    USER QUESTION: 
    {question}

    FINAL RESPONSE:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = prompt | llm
    
    try:
        response = chain.invoke({"context": context, "question": question})
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"
