import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import re
import joblib
import os
import base64
from PyPDF2 import PdfReader
from geopy.distance import distance
from geopy.geocoders import Nominatim
from functools import lru_cache
import concurrent.futures
import asyncio

# ===========================
# Load Model and Preprocessing Tools
# ===========================

model = tf.keras.models.load_model('talent_acquisition_model.h5')
tokenizer = joblib.load('tokenizer.pkl')
ner_pipeline = pipeline('ner', model='dslim/bert-base-NER')
geolocator = Nominatim(user_agent="talent_acquisition_app")

# ===========================
# Web Scraping for Fortune 500 and Top Universities
# ===========================

@st.cache_data
def get_fortune_500():
    response = requests.get('https://fortune.com/fortune500/')
    soup = BeautifulSoup(response.text, 'html.parser')
    return [tag.text.strip() for tag in soup.select('h3.company-title')]

@st.cache_data
def get_top_universities():
    response = requests.get('https://www.topuniversities.com/university-rankings/world-university-rankings/2024')
    soup = BeautifulSoup(response.text, 'html.parser')
    return [tag.text.strip() for tag in soup.select('div.uni-name')]

fortune_500_companies = get_fortune_500()
top_universities = get_top_universities()

# ===========================
# Optimized Helper Functions
# ===========================

@lru_cache(maxsize=1000)
def get_coordinates(city):
    try:
        location = geolocator.geocode(city)
        return (location.latitude, location.longitude) if location else None
    except:
        return None

def is_location_nearby(resume, target_location):
    cities = re.findall(r'\b([A-Za-z\s]+)\b', resume)
    target_coords = get_coordinates(target_location)
    if not target_coords:
        return 0
    for city in cities:
        city_coords = get_coordinates(city)
        if city_coords and distance(target_coords, city_coords).km <= 50:
            return 3
    return 0

def extract_skills(resume):
    ner_results = ner_pipeline(resume)
    return set(result['word'] for result in ner_results if result['entity'] == 'MISC')

def match_skills(resume_skills, job_skills):
    return len(set(resume_skills) & set(job_skills))

def is_fortune_500(resume):
    return any(company.lower() in resume.lower() for company in fortune_500_companies)

def is_top_university(resume):
    return any(uni.lower() in resume.lower() for uni in top_universities)

def is_referred(resume):
    return 'referral' in resume.lower() or 'referred by' in resume.lower()

def extract_years_experience(resume):
    match = re.search(r'(\d+)\s+years?', resume, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        return "".join([page.extract_text() or "" for page in pdf_reader.pages])
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def create_download_link(pdf_file_name):
    try:
        with open(pdf_file_name, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            b64 = base64.b64encode(pdf_bytes).decode()
            return f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_file_name}">Download Resume</a>'
    except FileNotFoundError:
        return "File not found"

def color_score(val):
    if val >= 8:
        return 'background-color: #4CAF50; color: white;'
    elif 5 <= val < 8:
        return 'background-color: #FFC107; color: black;'
    return 'background-color: #F44336; color: white;'

async def process_resume(file):
    text = extract_text_from_pdf(file)
    return {
        'Resume_str': text,
        'pdf_file_name': file.name,
        'skills_match': match_skills(extract_skills(text), []),
        'location_nearby': is_location_nearby(text, ""),
        'fortune_500': int(is_fortune_500(text)),
        'top_university': int(is_top_university(text)),
        'referral': int(is_referred(text)),
        'years_experience': extract_years_experience(text)
    }

# ===========================
# Streamlit Navigation
# ===========================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Candidate Ranking", "Upload Resumes", "Download Uploaded Resumes"])

# ===========================
# Page 1: Candidate Ranking
# ===========================

if page == "Candidate Ranking":
    st.title('Candidate Ranking')
    job_description = st.text_area("Enter Job Description")
    target_location = st.text_input("Enter Office Location")

    if os.path.exists('uploaded_resumes.csv'):
        df = pd.read_csv('uploaded_resumes.csv')

        if 'pdf_file_name' not in df.columns:
            st.error("PDF file names missing. Ensure resumes are uploaded with filenames!")

        if st.button('Rank Candidates'):
            job_skills = list(extract_skills(job_description))
            
            df['Score'] = model.predict(
                [pad_sequences(tokenizer.texts_to_sequences(df['Resume_str'].values), maxlen=300, padding='post', truncating='post'),
                 df[['skills_match', 'location_nearby', 'fortune_500', 'top_university', 'referral', 'years_experience']].values],
                batch_size=32
            ).round(1)

            styled_df = df[['Resume_str', 'Score', 'Download Resume']].sort_values(by='Score', ascending=False)
            st.write("### Candidate Scores (1-10) with Resume Downloads")
            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    else:
        st.warning("No resumes uploaded. Please go to 'Upload Resumes' page.")

# ===========================
# Page 2: Upload Resumes
# ===========================

elif page == "Upload Resumes":
    st.title('Upload Resumes')
    uploaded_files = st.file_uploader("Upload PDF Resumes", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        resumes = [{'Resume_str': extract_text_from_pdf(file), 'pdf_file_name': file.name} for file in uploaded_files]
        new_df = pd.DataFrame(resumes)

        if os.path.exists('uploaded_resumes.csv'):
            existing_df = pd.read_csv('uploaded_resumes.csv')
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates().reset_index(drop=True)
        else:
            combined_df = new_df

        combined_df.to_csv('uploaded_resumes.csv', index=False)
        st.success("Resumes processed, duplicates removed!")

# ===========================
# Page 3: Download Uploaded Resumes
# ===========================

elif page == "Download Uploaded Resumes":
    st.title('Download Uploaded Resumes')
    if os.path.exists('uploaded_resumes.csv'):
        with open('uploaded_resumes.csv', 'rb') as file:
