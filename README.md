#  End-to-End AI HR Automation System

An automated AI-powered Human Resources system designed to streamline the candidate evaluation process. The system uses advanced natural language processing and computer vision techniques to read CVs (Resumes), analyze them against specific Job Descriptions, grade candidates, and facilitate candidate querying.

##  Key Features

1. **CV Processing Pipeline**
   - Supports uploading multiple CVs in various formats (`.pdf`, `.png`, `.jpg`).
   - Automatically extracts text from standard PDFs.
   - Utilizes **EasyOCR** and **OpenCV** to extract text from scanned PDFs and Images, drawing boundary boxes for transparency.
2. **AI-Powered Evaluation**
   - Powered by **Groq API** (`llama-3.3-70b-versatile`) and **LangChain**.
   - Compares the extracted CV text against a provided Job Description.
   - Generates a **Match Score (0-100)**, extracts applicant details (name, email, skills, experience), and makes an initial **"Accepted / Rejected"** decision.
   - Automatically drafts an email to the candidate in their native language based on the final decision.
3. **HR Tracking Dashboard**
   - Saves all candidate metrics and statuses inside a localized `SQLite` Database.
   - Displays a dynamic ranking dashboard where you can track candidates by score.
   - Allows exporting candidate tabular data directly to a `.csv` file.
4. **Interactive AI Chatbot**
   - Query individual candidates using an integrated Chatbot.
   - The Chatbot retrieves the candidate's exact background context and can answer contextual questions about their experience.

##  Project Architecture

The project has been refactored into a scalable modular architecture:

- `app.py`: The lightweight main entry point of the Streamlit application.
- `ui.py`: Manages the frontend UI (tabs, buttons, metrics) rendered via Streamlit.
- `ai_agent.py`: Houses the LangChain pipeline to interact with the LLM.
- `extractor.py`: Manages document handling, PyPDF operations, and the EasyOCR image processing.
- `database.py`: Controls data flowing into and out of the `hr_database.db` SQLite database.

##  Installation & Setup

1. **Clone or Download the Repository:**
   Navigate to the project folder.

2. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Other implicit dependencies include `fitz` (PyMuPDF) and OpenCV (`cv2`), so ensure they are installed if encountering issues.*

3. **Set Up the Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY="your_api_key_here"
   ```

4. **Run the Application:**
   Start the Streamlit application by running:
   ```bash
   streamlit run app.py
   ```

##  Tech Stack
- **Frontend / Web UI**: [Streamlit](https://streamlit.io/)
- **LLM Engine**: [Groq](https://groq.com/) API / Llama 3.3
- **AI Orchestration**: [LangChain](https://python.langchain.com/)
- **Data & Storage**: SQLite & [Pandas](https://pandas.pydata.org/)
- **OCR & Vision**: [EasyOCR](https://github.com/JaidedAI/EasyOCR), OpenCV, Pillow
- **PDF Extraction**: `pypdf`, `PyMuPDF` (fitz)
