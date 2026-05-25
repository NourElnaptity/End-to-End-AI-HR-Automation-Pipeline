import os
import streamlit as st
from dotenv import load_dotenv

from database import init_db
from ui import render_ui

# Load environment variables
load_dotenv(override=True)

# Check for API Key
if not os.environ.get("GROQ_API_KEY"):
    st.error("Please set the GROQ_API_KEY environment variable in a .env file.")
    st.stop()

# Initialize DB if not exists
init_db()

# Render the Streamlit UI
if __name__ == "__main__":
    render_ui()
