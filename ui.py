import streamlit as st
from PIL import Image
import pandas as pd
from langchain_groq import ChatGroq

from database import save_to_db, load_db_data
from extractor import load_ocr_model, extract_text_and_boxes_from_image, extract_text_from_pdf
from ai_agent import analyze_cv_with_jd

def render_ui():
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
