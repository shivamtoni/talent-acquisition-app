import streamlit as st
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fuzzywuzzy import fuzz

# Load trained model
model = tf.keras.models.load_model('talent_acquisition_model.h5')

# Sample Fortune 500 and top universities lists
fortune_500_companies = ['Google', 'Amazon', 'Microsoft', 'Apple', 'Facebook']
top_universities = ['Harvard', 'MIT', 'Stanford', 'Oxford', 'Cambridge']

# Feature extraction functions
def extract_years_experience(resume):
    years = re.findall(r'(\d+)\s+years?', resume, re.IGNORECASE)
    return max(map(int, years), default=0)

def match_skills(resume, job_skills):
    resume_skills = re.findall(r'\b(Java|Python|TensorFlow|SQL|React|ML|AI)\b', resume, re.IGNORECASE)
    return len(set(resume_skills) & set(job_skills))

def is_fortune_500(resume):
    return int(any(company.lower() in resume.lower() for company in fortune_500_companies))

def is_top_university(resume):
    return int(any(uni.lower() in resume.lower() for uni in top_universities))

def is_referred(resume):
    return int('referral' in resume.lower() or 'referred by' in resume.lower())

def is_location_nearby(resume, target_location='San Francisco'):
    cities = re.findall(r'\b([A-Za-z\s]+)\b', resume)
    return int(any(fuzz.partial_ratio(city.lower(), target_location.lower()) > 80 for city in cities))

# Feature extraction
def extract_features(df, job_skills, target_location='San Francisco'):
    df['skills_match'] = df['Resume_str'].apply(lambda x: match_skills(x, job_skills) * 3)
    df['years_experience'] = df['Resume_str'].apply(extract_years_experience)
    df['fortune_500'] = df['Resume_str'].apply(is_fortune_500)
    df['location_nearby'] = df['Resume_str'].apply(lambda x: is_location_nearby(x, target_location) * 3)
    df['top_university'] = df['Resume_str'].apply(is_top_university) * 3
    df['referral'] = df['Resume_str'].apply(is_referred)
    return df

# Streamlit UI
st.title('Talent Acquisition Candidate Ranking')

job_description = st.text_area('Enter Job Description:')
target_location = st.text_input('Enter Office Location:', 'San Francisco')

job_skills = re.findall(r'\b(Java|Python|TensorFlow|SQL|React|ML|AI)\b', job_description, re.IGNORECASE)
st.write(f"Extracted Job Skills: {job_skills}")

uploaded_file = st.file_uploader('Upload Resume Dataset (CSV)', type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = extract_features(df, job_skills, target_location)

    # Tokenization and padding
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['Resume_str'].values)
    sequences = tokenizer.texts_to_sequences(df['Resume_str'].values)
    padded = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')

    additional_features = df[['skills_match', 'years_experience', 'fortune_500', 'location_nearby', 'top_university', 'referral']].values

    # Model predictions
    predictions = model.predict([padded, additional_features])
    df['score'] = predictions
    ranked_candidates = df.sort_values(by='score', ascending=False)

    st.subheader('Ranked Candidates')
    st.dataframe(ranked_candidates[['Resume_str', 'score']])
