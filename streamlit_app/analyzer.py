import re
import math
import json
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# --- Preprocessing ---
def preprocess_text(text: str) -> str:
    text = text or ""
    text = text.replace('\n', ' ').strip()
    text = re.sub(r"\s+", ' ', text)
    text = text.lower()
    return text

def tokenize_and_lemmatize(text: str):
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens

# --- Readability (Flesch Reading Ease) ---
def count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    syllables = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            syllables += 1
        prev_was_vowel = is_vowel
    if word.endswith('e') and syllables > 1:
        syllables -= 1
    return max(1, syllables)

def flesch_reading_ease(text: str) -> float:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = []
    syllable_count = 0
    for s in sentences:
        ws = [w for w in re.findall(r"[A-Za-z]+", s)]
        words.extend(ws)
        for w in ws:
            syllable_count += count_syllables(w)
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    asl = num_words / num_sentences  # average sentence length
    asw = syllable_count / num_words  # avg syllables per word
    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return round(score, 2)

# --- Keyword extraction from JD ---
def jd_top_keywords(jd_text: str, top_n: int = 20):
    jd_text = preprocess_text(jd_text)
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    tfidf = vec.fit_transform([jd_text])
    scores = zip(vec.get_feature_names_out(), tfidf.toarray()[0])
    sorted_feats = sorted(scores, key=lambda x: x[1], reverse=True)
    top = [feat for feat,score in sorted_feats[:top_n] if feat.isalpha() or ' ' in feat]
    return top

# --- Skill overlap & density ---
def compute_keyword_density(resume_text: str, keywords: list) -> dict:
    resume_text = preprocess_text(resume_text)
    words = re.findall(r"[a-z]+", resume_text)
    total_words = len(words) or 1
    counts = Counter()
    for k in keywords:
        counts[k] = len(re.findall(r"\b" + re.escape(k.lower()) + r"\b", resume_text))
    density_per_1000 = {k: round((counts[k] / total_words) * 1000, 3) for k in keywords}
    return {
        'counts': dict(counts),
        'density_per_1000': density_per_1000,
        'total_words': total_words
    }

# --- Skill relevance (TF-IDF cosine between resume and JD) ---
def skill_relevance_score(resume_text: str, jd_text: str) -> float:
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    X = vec.fit_transform([preprocess_text(resume_text), preprocess_text(jd_text)])
    sim = cosine_similarity(X[0:1], X[1:2])[0][0]
    return round(sim * 100, 2)

# --- ATS heuristic suggestions (simple rule-based) ---
def ats_suggestions(resume_text: str, jd_keywords: list) -> list:
    suggestions = []
    text = preprocess_text(resume_text)
    missing = [k for k in jd_keywords if re.search(r"\b"+re.escape(k)+r"\b", text) is None]
    if missing:
        suggestions.append({'issue': 'missing_keywords', 'detail': missing[:10],
                            'suggestion': 'Add these top JD keywords into Skills/Experience where relevant.'})
    sentences = re.split(r'[.!?]+', resume_text)
    long_sentences = [s for s in sentences if len(s.split()) > 40]
    if long_sentences:
        suggestions.append({'issue': 'long_sentences', 'count': len(long_sentences),
                            'suggestion': 'Shorten long sentences; use concise bullets (<= 30 words).'} )
    kd = compute_keyword_density(resume_text, jd_keywords)
    avg_density = sum(kd['density_per_1000'].values())/ (len(jd_keywords) or 1)
    if avg_density < 0.5:
        suggestions.append({'issue': 'low_keyword_density', 'value': round(avg_density,3),
                            'suggestion': 'Increase presence of role-specific keywords in Experience and Skills sections.'})
    return suggestions

# --- Composite scoring ---
def compute_scores(resume_text: str, jd_text: str):
    skill_score = skill_relevance_score(resume_text, jd_text)
    flesch = flesch_reading_ease(resume_text)
    readability_score = max(0.0, min(100.0, flesch))
    jd_keywords = jd_top_keywords(jd_text, top_n=20)
    kd = compute_keyword_density(resume_text, jd_keywords)
    avg_density = sum(kd['density_per_1000'].values()) / (len(jd_keywords) or 1)
    density_score = max(0.0, min(100.0, (avg_density / 10.0) * 100.0))
    total = round((0.6 * skill_score) + (0.2 * readability_score) + (0.2 * density_score), 2)
    suggestions = ats_suggestions(resume_text, jd_keywords)
    return {
        'skill_score': skill_score,
        'readability_score': readability_score,
        'density_score': density_score,
        'total_score': total,
        'flesch': flesch,
        'jd_keywords': jd_keywords,
        'keyword_counts': kd['counts'],
        'keyword_density_per_1000': kd['density_per_1000'],
        'suggestions': suggestions
    }

if __name__ == '__main__':
    sample_resume = """Experienced software engineer with experience in python, machine learning, scikit-learn. Built REST API."""
    sample_jd = """Looking for a Python developer with experience in scikit-learn, pandas, machine learning, REST API development."""
    print(json.dumps(compute_scores(sample_resume, sample_jd), indent=2))
