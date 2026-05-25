# End-to-End AI HR Automation Pipeline

## Project Objective
The main goal of this project is to automate the hiring and CV screening workflow using Large Language Models (LLMs) and Computer Vision techniques. The system is designed to seamlessly process multi-lingual resumes, evaluate candidates against specific Job Descriptions, and handle communication automatically to reduce manual HR workloads and human bias.

The system is designed to provide production-ready document processing, intelligent AI agent decision-making, and persistent data tracking.

### Extracted Information & Actions:
* Candidate Metadata: Full Name, Contact Info, and Core Technical Skills.
* Match Score: Algorithmic calculation of candidate compatibility with the Job Description.
* Hiring Decision: Automated dynamic acceptance or rejection status based on JSON criteria.
* Automated Emailing: Generation of context-aware, personalized feedback emails for applicants.
* Interactive Dashboard: Full visualization of processed data, charts, and real-time database queries.

---

## System Requirements
* Python 3.10.x or 3.11.x (Mandatory)

> Important Note:
> This project utilizes advanced AI Agent frameworks and specific multi-lingual OCR dependencies. Ensure your environment matches the Python versions specified to avoid package compatibility issues with deep learning backends.

---

## Installation & Setup (Step by Step)

# 1. Create a Virtual Environment
Create a clean isolated virtual environment for the project dependencies:
```bash
py -3.10 -m venv hrenv
```

# 2. Activate the Environment
Activate the environment to start working:
```bash
hrenv\Scripts\activate
```

# 3. Install Core AI Dependencies (CPU Version)
The project runs on CPU only (no GPU required):
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

# 4. Install Project Dependencies
All required libraries are listed in requirements.txt:
```bash
pip install -r requirements.txt
```

# 5. Running the Project
Start the Streamlit Application:
```bash
streamlit run app.py
```
