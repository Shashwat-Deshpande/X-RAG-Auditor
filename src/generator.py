from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def generate_answer(question, context, api_key):
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0, 
        groq_api_key=api_key
    )
    
    # We remove the "STOP ALL ANALYSIS" hard-block to prevent false negatives
    template = """
    ROLE: You are an Expert Insurance Auditor.
    
    INSTRUCTIONS:
    1. Look for keywords like "Exclusion", "Waiting Period", "Sum Insured", or "Policy" in the context.
    2. If the context is clearly unrelated to insurance (e.g. a recipe or math), say: "This document does not seem to contain insurance rules."
    3. If it IS insurance, answer the question accurately.
    4. For the Glaucoma/Surgery question: Check the 'Specific Waiting Period' table.
    5. Cite the exact page and any 'Excl' codes found.

    CONTEXT: 
    {context}

    USER QUESTION: 
    {question}

    FINAL AUDIT REPORT:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = prompt | llm
    
    try:
        response = chain.invoke({"context": context, "question": question})
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"
