import sqlite3
import pandas as pd

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
