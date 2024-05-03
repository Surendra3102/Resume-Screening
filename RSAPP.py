import streamlit as st
import PyPDF2
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration and background color
st.set_page_config(page_title="Candidate Selection Tool", page_icon=":clipboard:", layout="centered", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to display error message
def show_error_message(message):
    st.error(message)

# Function to display success message
def show_success_message(message):
    st.success(message)

# Function to display loading message
def show_loading_message(message):
    st.info(message)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except:
        return None
    return text

# Function to calculate match percentage
def calculate_match_percentage(job_description, resume):
    content = [job_description, resume]
    cv = CountVectorizer()
    matrix = cv.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    match_percentage = similarity_matrix[0][1] * 100
    return round(match_percentage, 2)

# Title and description
st.title("Candidate Selection Tool")
st.subheader("NLP Based Resume Screening")
st.write("Aim of this project is to check whether a candidate is qualified for a role based on their education, experience, and other information captured on their resume. In a nutshell, it's a form of pattern matching between a job's requirements and the qualifications of a candidate based on their resume.")

# File uploaders and process button
uploadedJD = st.file_uploader("Upload Job Description (PDF)", type="pdf")
uploadedResume = st.file_uploader("Upload Resume (PDF)", type="pdf")
click = st.button("Process")

if click:
    if uploadedJD is None or uploadedResume is None:
        show_error_message("Please upload both Job Description and Resume to proceed.")
    else:
        show_loading_message("Processing... This may take a moment.")
        job_description = extract_text_from_pdf(uploadedJD)
        resume = extract_text_from_pdf(uploadedResume)
        
        if job_description is None or resume is None:
            show_error_message("Error reading PDF file. Please ensure the files are valid PDFs.")
        else:
            match_percentage = calculate_match_percentage(job_description, resume)
            show_success_message(f"Match Percentage: {match_percentage}%")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Made by AI STARS")
