import streamlit as st
import pandas as pd
from google import genai
from core_logic import extract_text, check_plagiarism, grade_submission

st.set_page_config(page_title="AI Assignment Grader", layout="wide")
st.title("Automated Assignment & Project Report Grader")

with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    max_marks = st.number_input("Maximum Marks", min_value=5, max_value=100, value=20)
    rubric = st.text_area(
        "Evaluation Rubric:",
        value="1. Problem Definition & Abstract (5 marks)\n2. Methodology & Technical Depth (8 marks)\n3. Results & Conclusion (4 marks)\n4. Clarity & Structure (3 marks)",
        height=180
    )

st.header("2. Upload Student Reports (PDF / DOCX)")
uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Evaluate Submissions", type="primary"):
    if not api_key:
        st.error("Sidebar mein apni Gemini API Key daalein!")
    elif not uploaded_files:
        st.warning("Kam se kam ek PDF ya DOCX file upload karein!")
    else:
        student_texts = {}
        results = []
        with st.spinner("Evaluating reports with AI..."):
            client = genai.Client(api_key=api_key)
            for file in uploaded_files:
                text = extract_text(file,client)
                student_texts[file.name] = text
                grade_data = grade_submission(api_key, text, rubric, max_marks)
                results.append({
                    "Student File": file.name,
                    "Score": grade_data["score"],
                    "Max Marks": max_marks,
                    "Strengths": grade_data["strengths"],
                    "Missing Points": grade_data["missing_points"],
                    "Viva Questions": " | ".join(grade_data["viva_questions"])
                })
        
        st.subheader("📊 Class Marksheet")
        df = pd.DataFrame(results)
        st.dataframe(df[["Student File", "Score", "Max Marks", "Strengths", "Missing Points"]], use_container_width=True)
        
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Marksheet (CSV)", data=csv, file_name="marksheet.csv", mime="text/csv")
        
        st.subheader("🚨 Cross-Plagiarism Check")
        alerts = check_plagiarism(student_texts)
        if alerts:
            st.error("High similarity detected!")
            st.table(pd.DataFrame(alerts))
        else:
            st.success("No copy-paste detected.")
            
        st.subheader("🎤 Auto-Generated Viva Questions")
        for res in results:
            with st.expander(f"Viva Questions: {res['Student File']} ({res['Score']}/{max_marks})"):
                for q in res["Viva Questions"].split(" | "):
                    st.write(f"- {q}")
