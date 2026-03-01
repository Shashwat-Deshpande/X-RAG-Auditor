from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def generate_answer(question, context, api_key):
    # Keeping temperature at 0 for strict reliability
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.1, 
        groq_api_key=api_key
    )
    
    template = """
    YOU ARE A SENIOR INSURANCE CLAIMS AUDITOR. Your goal is to find the truth, even if it's hidden in tables.

    HIERARCHY RULES:
    1. CATEGORY MATCHING: If a specific surgery (e.g., Joint Replacement) isn't mentioned as a header, look for it in lists under "Specified Illnesses" or "Modern Treatments".
    2. TABLE LOGIC: If you see a list of surgeries followed by a list of months (12, 24, 36), the number applies to every surgery in that section.
    3. EXCLUSION CODES: Look for codes like (Excl01, Excl02). If a claim falls under an Exclusion code, it is REJECTED regardless of other benefits.
    4. CITE EVERYTHING: Use [Page X] for every rule found.
    5. BE CONCISE. Do not repeat the same rule more than once.
    6. Distinguish between In-Patient (Hospitalization) and Out-Patient (OPD) rules based on the surgery type.
    7. Distinguish between In-Patient and Out-Patient rules. A surgery (Hospitalization) should NOT be rejected based on Out-Patient (OPD) waiting periods unless specifically linked.

    MANDATORY OUTPUT STRUCTURE (FOLLOW THIS EVERY TIME):
    - **HOSPITALIZATION TYPE**: [State if this is In-Patient or Out-Patient and why]
    - **ELIGIBILITY CHECK**: [Check Age and Policy Tenure vs Policy limits]
    - **SURGERY & WAITING PERIOD**: [Identify the surgery category and the applicable waiting period (e.g., 24/36 months)]
    - **EXCLUSIONS**: [State if any Exclusion Codes apply]
    - **FINAL AUDIT VERDICT**: [APPROVED/REJECTED with 1-sentence reason]

    CONTEXT: {context}
    QUESTION: {question}

    FINAL AUDIT REPORT:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = prompt | llm
    
    try:
        response = chain.invoke({"context": context, "question": question})
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"