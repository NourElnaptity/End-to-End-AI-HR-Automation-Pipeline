from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def analyze_cv_with_jd(text, jd):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="""You are an expert HR AI Agent.
        Analyze the following CV/Resume text against the provided Job Description.
        
        Job Description:
        {jd}
        
        CV Text:
        {cv_text}
        
        Step 1: Detect language (ar or en).
        Step 2: Extract Name, Email, Skills, Experience (short summary).
        Step 3: Generate a Match Score (0 to 100 integer) based on how well the CV fits the Job Description.
        Step 4: Make a Decision ("Accepted" if score >= 70, otherwise "Rejected").
        Step 5: Write a short Evaluation explaining the score.
        Step 6: Write an email draft in the candidate's language informing them of the decision.
        
        Output JSON exactly matching:
        {{
            "language": "ar or en",
            "name": "Candidate Name",
            "email": "Candidate Email",
            "skills": ["skill1", "skill2"],
            "experience": "Brief experience summary",
            "score": 85,
            "decision": "Accepted or Rejected",
            "evaluation": "Evaluation text...",
            "email_draft": "Email text in appropriate language..."
        }}
        """,
        input_variables=["jd", "cv_text"]
    )
    chain = prompt | llm | parser
    return chain.invoke({"jd": jd, "cv_text": text})
