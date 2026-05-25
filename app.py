import streamlit as st
import os
import sqlite3
import easyocr
import pypdf
import fitz  # PyMuPDF
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser

import io

# Load environment variables
load_dotenv(override=True)

# Check for API Key
if not os.environ.get("GROQ_API_KEY"):
    st.error("Please set the GROQ_API_KEY environment variable in a .env file.")
    st.stop()

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('hr_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            skills TEXT,
            experience TEXT,
            language TEXT,
            evaluation TEXT,
            score INTEGER,
            decision TEXT,
            job_description TEXT,
            status TEXT,
            raw_text TEXT
        )
    ''')
    # Try adding new columns if old DB exists
    try: cursor.execute('ALTER TABLE candidates ADD COLUMN experience TEXT')
    except: pass
    try: cursor.execute('ALTER TABLE candidates ADD COLUMN score INTEGER')
    except: pass
    try: cursor.execute('ALTER TABLE candidates ADD COLUMN decision TEXT')
    except: pass
    try: cursor.execute('ALTER TABLE candidates ADD COLUMN job_description TEXT')
    except: pass
    try: cursor.execute('ALTER TABLE candidates ADD COLUMN raw_text TEXT')
    except: pass
    conn.commit()
    conn.close()

def save_to_db(data, jd, raw_text):
    conn = sqlite3.connect('hr_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO candidates (name, email, skills, experience, language, evaluation, score, decision, job_description, status, raw_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (data.get('name', 'N/A'), data.get('email', 'N/A'), 
          ', '.join(data.get('skills', [])), data.get('experience', 'N/A'),
          data.get('language', 'en'), data.get('evaluation', 'N/A'), 
          data.get('score', 0), data.get('decision', 'Pending'), jd, 'Processed', raw_text))
    conn.commit()
    conn.close()

def load_db_data():
    conn = sqlite3.connect('hr_database.db')
    df = pd.read_sql_query("SELECT * FROM candidates", conn)
    conn.close()
    return df

init_db()

# --- OCR & Text Extraction Setup ---
@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['ar', 'en'], gpu=False)

def extract_text_and_boxes_from_image(image, reader):
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    results = reader.readtext(image_np)
    text = " ".join([res[1] for res in results])
    
    # Draw boxes
    for (bbox, t, prob) in results:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        cv2.rectangle(image_np, tl, br, (0, 255, 0), 2)
        
    return text, Image.fromarray(image_np)

def extract_text_from_pdf(pdf_file, reader):
    reader_pdf = pypdf.PdfReader(pdf_file)
    text = ""
    for page in reader_pdf.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
            
    images_with_boxes = []
    
    if len(text.strip()) < 50:
        # Scanned PDF logic
        st.info("PDF appears to be scanned. Extracting using OCR...")
        pdf_file.seek(0)
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text, img_boxed = extract_text_and_boxes_from_image(img, reader)
            text += page_text + "\n"
            images_with_boxes.append(img_boxed)
            
    return text, images_with_boxes

# --- AI Agent Setup ---
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



# --- Streamlit UI ---
st.set_page_config(page_title="End-to-End AI HR Automation", page_icon="🏢", layout="wide")
st.title("🏢 End-to-End AI HR Automation System")
st.write("Complete automated pipeline for processing, ranking, and interacting with candidate profiles.")

tab1, tab2, tab3 = st.tabs(["🚀 1. CV Processing Pipeline", "📊 2. HR Dashboard", "💬 3. AI Chatbot"])

# --- TAB 1: Processing ---
with tab1:
    col_jd, col_upload = st.columns([1, 1])
    
    with col_jd:
        st.subheader("📝 Job Description")
        job_description = st.text_area("Enter the Job Description for matching:", height=200, 
                                       placeholder="e.g. Seeking a Senior Python Developer with 5+ years of experience in Django, React, and AWS...")
                                       
    with col_upload:
        st.subheader("📄 Upload CVs")
        uploaded_files = st.file_uploader("Upload CVs (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

    if st.button("Process All Candidates", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one CV.")
        elif not job_description:
            st.warning("Please enter a Job Description.")
        else:
            reader = load_ocr_model()
            
            for file in uploaded_files:
                st.markdown(f"### Processing: `{file.name}`")
                
                text = ""
                images_with_boxes = []
                
                with st.spinner("Extracting text & applying OCR (if needed)..."):
                    if file.name.lower().endswith('.pdf'):
                        text, images_with_boxes = extract_text_from_pdf(file, reader)
                    else:
                        img = Image.open(file)
                        text, boxed_img = extract_text_and_boxes_from_image(img, reader)
                        images_with_boxes.append(boxed_img)
                
                if images_with_boxes:
                    with st.expander("Show OCR Bounding Boxes Visualizations"):
                        for boxed_img in images_with_boxes:
                            st.image(boxed_img, use_column_width=True)
                
                if text.strip() == "":
                    st.error(f"Failed to extract text from {file.name}.")
                    continue
                
                with st.spinner("AI Analysis & Job Matching in progress..."):
                    try:
                        result = analyze_cv_with_jd(text, job_description)
                        st.success(f"Analysis Complete for {result.get('name')}!")
                        
                        score = result.get('score', 0)
                        decision = result.get('decision', 'Pending')
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        metric_col1.metric("Match Score", f"{score}/100")
                        metric_col2.metric("Decision", decision)
                        metric_col3.metric("Detected Language", result.get('language'))
                        
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.write("**Name:**", result.get('name'))
                            st.write("**Email:**", result.get('email'))
                            st.write("**Skills:**", ", ".join(result.get('skills', [])))
                            st.write("**Experience:**", result.get('experience'))
                            st.write("**AI Evaluation:**", result.get('evaluation'))
                        
                        with res_col2:
                            st.write("**Automated Email Draft:**")
                            st.info(result.get('email_draft'))
                        
                        save_to_db(result, job_description, text)
                        
                    except Exception as e:
                        st.error(f"Error analyzing {file.name}: {e}")
                st.divider()
            
            st.success("🎉 All candidates processed successfully! Check the HR Dashboard.")

# --- TAB 2: HR Dashboard ---
with tab2:
    st.subheader("📊 Candidate Tracking & Ranking")
    st.write("View all processed candidates, ranked by their Match Score.")
    
    if st.button("Refresh Dashboard"):
        st.rerun()
        
    df = load_db_data()
    if not df.empty:
        # Rank by score descending
        df_ranked = df.sort_values(by="score", ascending=False).reset_index(drop=True)
        
        # Display as interactive dataframe
        st.dataframe(
            df_ranked[['name', 'score', 'decision', 'email', 'skills', 'experience', 'language']],
            use_container_width=True,
            hide_index=True
        )
        
        # CSV Export
        csv = df_ranked.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Automated Report (CSV)",
            data=csv,
            file_name='hr_candidates_report.csv',
            mime='text/csv',
        )
    else:
        st.info("No candidates processed yet.")

# --- TAB 3: AI Chatbot ---
with tab3:
    st.subheader("💬 Ask AI about a Candidate")
    
    df = load_db_data()
    if not df.empty:
        candidate_names = df['name'].tolist()
        selected_candidate = st.selectbox("Select a Candidate:", ["Choose a candidate..."] + candidate_names)
        
        if selected_candidate != "Choose a candidate...":
            candidate_data = df[df['name'] == selected_candidate].iloc[0]
            
            # Initialize session state for chatbot
            chat_key = f"chat_{selected_candidate}"
            if chat_key not in st.session_state:
                # Provide candidate context to the AI
                context = f"""You are an HR Assistant. You are answering questions about the following candidate.
                Name: {candidate_data['name']}
                Skills: {candidate_data['skills']}
                Experience: {candidate_data['experience']}
                Raw CV Text: {candidate_data['raw_text']}
                """
                st.session_state[chat_key] = [
                    ("system", context),
                    ("assistant", "Understood. I am ready to answer questions about this candidate.")
                ]

            # Display chat history (skip system message)
            for role, content in st.session_state[chat_key]:
                if role == "system":
                    continue
                with st.chat_message(role):
                    st.markdown(content)

            # Chat input
            if prompt := st.chat_input("Ask a question about this candidate..."):
                st.session_state[chat_key].append(("user", prompt))
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
                    response = llm.invoke(st.session_state[chat_key])
                    answer = response.content
                    st.markdown(answer)
                    st.session_state[chat_key].append(("assistant", answer))
    else:
        st.info("Process some candidates first before using the chatbot.")
