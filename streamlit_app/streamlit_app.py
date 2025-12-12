# streamlit_app/streamlit_app.py
import streamlit as st
import json
import sys
import os
sys.path.append(os.path.abspath('../backend'))
from analyzer import compute_scores

from PyPDF2 import PdfReader
import io

st.title('Resume Analyzer — Streamlit demo')

resume_input = st.text_area('Paste resume text here', height=200)
uploaded_file = st.file_uploader("Or upload a PDF resume", type=['pdf'])
jd_input = st.text_area('Paste job description (JD) here', height=200)

resume_text = None
if uploaded_file is not None:
    try:
        # read uploaded file bytes and extract text
        bytes_data = uploaded_file.read()
        reader = PdfReader(io.BytesIO(bytes_data))
        pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt)
        resume_text = "\n".join(pages).strip()
        if not resume_text:
            st.error("No text was extracted from the PDF. If this is a scanned PDF, OCR is required.")
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")

# choose final input text (uploaded pdf has priority)
final_resume_text = resume_text if resume_text else resume_input

if st.button('Analyze'):
    if not final_resume_text or not jd_input:
        st.warning('Please provide both resume text (paste or upload PDF) and a JD.')
    else:
        with st.spinner('Analyzing...'):
            result = compute_scores(final_resume_text, jd_input)
        st.subheader(f"Total score: {result['total_score']} / 100")
        st.write('Skill score:', result['skill_score'])
        st.write('Readability score:', result['readability_score'], '(Flesch =', result['flesch'],')')
        st.write('Keyword density score:', result['density_score'])

        st.subheader('Top JD keywords')
        st.write(result['jd_keywords'])

        st.subheader('Keyword counts')
        st.json(result['keyword_counts'])

        st.subheader('Suggestions (ATS & improvements)')
        st.json(result['suggestions'])
