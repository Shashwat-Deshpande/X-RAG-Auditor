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
    ROLE: Expert Insurance Auditor.
    
    CONTEXT ANALYSIS:
    The provided context contains multiple pages of an insurance policy. 
    You MUST search specifically for a table or list titled "Waiting Periods" or "Specific Waiting Period".
    Look for "Cataract", "Glaucoma", or "Eye surgery" within those lists.

    AUDIT STEPS:
    1. If you find a list of diseases with numbers like 12, 24, or 48 next to them, those are the waiting periods.
    2. Compare the 14-month claim date to that number.
    3. If the table is truly missing from the context, state "Table not found in current context."

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
