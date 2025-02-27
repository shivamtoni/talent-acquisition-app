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
from fuzzywuzzy import fuzz
import joblib
import os
from PyPDF2 import PdfReader

# Load the trained model
model = tf.keras.models.load_model('talent_acquisition_model.h5')
tokenizer = joblib.load('tokenizer.pkl')

# BERT NER pipeline
ner_pipeline = pipeline('ner', model='dslim/bert-base-NER')

# Web scraping for Fortune 500 and top universities
def get_fortune_500():
    response = requests.get('https://fortune.com/fortune500/')
    soup = BeautifulSoup(response.text, 'html.parser')
    return [tag.text.strip() for tag in soup.select('h3.company-title')]

def get_top_universities():
    response = requests.get('https://www.topuniversities.com/university-rankings/world-university-rankings/2024')
    soup = BeautifulSoup(response.text, 'html.parser')
    return [tag.text.strip() for tag in soup.select('div.uni-name')]

fortune_500_companies = get_fortune_500()
top_universities = get_top_universities()

# Feature extraction
def extract_skills(resume):
    ner_results = ner_pipeline(resume)
    return set(result['word'] for result in ner_results if result['entity'] == 'MISC')

def match_skills(resume, job_skills):
    resume_skills = extract_skills(resume)
    return len(set(resume_skills) & set(job_skills))

def is_fortune_500(resume):
    return any(company.lower() in resume.lower() for company in fortune_500_companies)

def is_top_university(resume):
    return any(uni.lower() in resume.lower() for uni in top_universities)

def is_referred(resume):
    return 'referral' in resume.lower() or 'referred by' in resume.lower()

def is_location_nearby(resume, target_location):
    cities = re.findall(r'\b([A-Za-z\s]+)\b', resume)
    return any(fuzz.partial_ratio(city.lower(), target_location.lower()) > 80 for city in cities)

# PDF to text extractor
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Streamlit Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Resumes", "Candidate Ranking"])

# Page 1: Upload Resumes
if page == "Upload Resumes":
    st.title('Upload Resumes')
    uploaded_files = st.file_uploader("Upload PDF Resumes", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        resumes = []
        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            resumes.append({'Resume_str': resume_text})
        df = pd.DataFrame(resumes)
        df.to_csv('uploaded_resumes.csv', index=False)
        st.success("Resumes have been processed and saved!")

# Page 2: Candidate Ranking
elif page == "Candidate Ranking":
    st.title('Candidate Ranking')
    job_description = st.text_area("Enter Job Description")
    target_location = st.text_input("Enter Office Location")
    
    if os.path.exists('uploaded_resumes.csv'):
        df = pd.read_csv('uploaded_resumes.csv')
        if st.button('Rank Candidates'):
            job_skills = list(extract_skills(job_description))

            df['skills_match'] = df['Resume_str'].apply(lambda x: match_skills(x, job_skills) * 3)
            df['location_nearby'] = df['Resume_str'].apply(lambda x: is_location_nearby(x, target_location) * 3)
            df['fortune_500'] = df['Resume_str'].apply(lambda x: int(is_fortune_500(x)))
            df['top_university'] = df['Resume_str'].apply(lambda x: int(is_top_university(x)) * 3)
            df['referral'] = df['Resume_str'].apply(lambda x: int(is_referred(x)))

            sequences = tokenizer.texts_to_sequences(df['Resume_str'].values)
            padded = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')
            additional_features = df[['skills_match', 'location_nearby', 'fortune_500', 'top_university', 'referral']].values

            scores = model.predict([padded, additional_features])
            df['Score'] = scores

            st.write("### Candidate Scores")
            st.dataframe(df[['Resume_str', 'Score']].sort_values(by='Score', ascending=False))
    else:
        st.warning("No resumes uploaded yet. Please go to the 'Upload Resumes' page.")
