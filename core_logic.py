import os
import json
import time
import pymupdf as fitz
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types


def get_candidate_models(client):
    available = [m.name.replace("models/", "") for m in client.models.list()]
    preferred = ["gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    ordered = [m for m in preferred if m in available]
    for m in available:
        if "flash" in m and "embedding" not in m and m not in ordered:
            ordered.append(m)
    return ordered or available[:3]


def generate_with_fallback(client, contents):
    models_to_try = get_candidate_models(client)
    last_error = None
    for model_name in models_to_try:
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
            )
        except Exception as e:
            last_error = e
            err_msg = str(e).lower()
            if any(code in err_msg for code in ["404", "not_found", "503", "429", "overloaded", "unavailable", "resource_exhausted", "high demand"]):
                time.sleep(1)
                continue
            raise e
    raise last_error


def extract_text(uploaded_file, client=None):
    filename = uploaded_file.name.lower()
    text = ""
    if filename.endswith(".pdf"):
        pdf_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()

        # Agar PDF scanned/handwritten hai (text empty aaya), toh Gemini se OCR karwayein
        if len(text.strip()) < 20 and client is not None:
            response = generate_with_fallback(
                client,
                [
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    "Extract and return all text from this document accurately without any extra commentary."
                ]
            )
            text = response.text or ""

    elif filename.endswith(".docx"):
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text.strip()


def check_plagiarism(student_texts: dict):
    names = list(student_texts.keys())
    texts = list(student_texts.values())
    if len(names) < 2:
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)

    alerts = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = round(sim_matrix[i][j] * 100, 2)
            if score >= 60.0:
                alerts.append({
                    "Student 1": names[i],
                    "Student 2": names[j],
                    "Similarity (%)": score
                })
    return alerts


def grade_submission(api_key, student_text, rubric, max_marks):
    client = genai.Client(api_key=api_key)
    models_to_try = get_candidate_models(client)

    prompt = f"""
    You are a fair university professor. Evaluate this student assignment based on the rubric.

    Rubric: {rubric}
    Maximum Marks: {max_marks}

    Student Submission:
    {student_text[:12000]}

    Return ONLY a valid JSON object with no markdown formatting:
    {{
        "score": <number out of {max_marks}>,
        "strengths": "<2 concise bullet points>",
        "missing_points": "<2 concise bullet points>",
        "viva_questions": ["<Question 1>", "<Question 2>", "<Question 3>"]
    }}
    """
    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            clean_json = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            return json.loads(clean_json)
        except Exception as e:
            last_error = e
            time.sleep(1)
    raise last_error