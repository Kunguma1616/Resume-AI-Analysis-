# app.py
import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import re
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path
import base64
import time
from typing import List, Optional
from pydantic import BaseModel, Field

# LangChain imports for structured output
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ---------- PDF (ReportLab) ----------
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable
)

# ============ STRUCTURED OUTPUT SCHEMA ============
class CriterionScore(BaseModel):
    """Individual criterion evaluation"""
    criterion_name: str = Field(description="Name of the evaluation criterion")
    score: float = Field(description="Score out of 5", ge=1.0, le=5.0)
    explanation: str = Field(description="Detailed explanation for the score")

class ResumeAnalysisOutput(BaseModel):
    """Structured output schema for resume analysis"""
    criteria_scores: List[CriterionScore] = Field(
        description="List of 5-7 key criteria with scores and explanations"
    )
    pros: List[str] = Field(description="Strengths of the resume based on job description")
    cons: List[str] = Field(description="Weaknesses or gaps in the resume")
    overall_assessment: str = Field(description="Overall match assessment with improvement suggestions")
    key_requirements_met: List[str] = Field(description="Key job requirements that are met")
    key_requirements_missing: List[str] = Field(description="Key job requirements that are missing")

# ============ ENV & MODEL ============
load_dotenv(find_dotenv())

api_key = None
try:
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass
if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ============ PAGE CONFIG ============
st.set_page_config(page_title="Aspect AI Resume Analyzer", page_icon="✅", layout="wide")

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()
LOGO_FILE = str(SCRIPT_DIR / "images.png")

# ============ CSS / THEME ============
st.markdown("""
<style>
    :root {
        --color-primary: #27549D;
        --color-dark-blue: #0f1e33;
        --color-secondary: #7099DB;
        --color-accent: #F1FF24;
        --text-light: #FFFFFF;
        --text-dark: #0f172a;
        --card-bg: rgba(255, 255, 255, 0.98);
        --glass-bg: rgba(255, 255, 255, 0.12);
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f1e33 0%, #27549D 100%) !important;
    }
    [data-testid="stAppViewContainer"] > .main, .block-container {
        background: transparent !important;
    }
    .stApp {
        color: var(--text-light) !important;
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans';
    }
    .hero-header {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
        display: flex; flex-direction: column; align-items: center; gap: 1rem;
    }
    .hero-header img { max-width: 150px; margin-bottom: 1rem; }
    .hero-header h1 { color: var(--color-accent); font-size: 3rem; font-weight: 900; margin: 0; }
    .hero-header p { color: rgba(255, 255, 255, 0.92); font-size: 1.15rem; margin-top: 0.5rem; }

    .stForm, .score-container, .report-container, .white-card {
        background: var(--card-bg) !important; color: var(--text-dark) !important;
        border-radius: 20px; padding: 2rem; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .report-container h2, .white-card h2, .stForm h2,
    .report-container h3, .white-card h3, .stForm h3 {
        color: var(--color-primary) !important;
    }

    .score-card {
        background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%) !important;
        color: #FFFFFF !important; padding: 2rem !important; border-radius: 15px !important;
        text-align: center !important; margin: 1rem 0 !important; box-shadow: 0 10px 25px rgba(39,84,157,0.4) !important;
    }
    .score-card h3 { margin: 0; font-size: 1.05rem; opacity: 0.9; }
    .score-card h2 { margin: 0.5rem 0 0 0; font-size: 3rem; font-weight: 900; color: var(--color-accent); }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important; color: white !important;
        border: none !important; border-radius: 50px !important; padding: 0.85rem 2.2rem !important;
        font-weight: 800 !important; box-shadow: 0 8px 20px rgba(34, 197, 94, 0.45) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============ HELPERS ============
def get_image_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.warning(f"Logo file not found at: {file}")
        return None

def extract_pdf_text(uploaded_file):
    try:
        return extract_text(uploaded_file)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

@st.cache_resource
def get_sentence_transformer_model():
    with st.spinner("Loading sentence model... (first time only)"):
        return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def calculate_similarity_bert(text1, text2):
    m = get_sentence_transformer_model()
    e1 = m.encode([text1]); e2 = m.encode([text2])
    return cosine_similarity(e1, e2)[0][0]

# ============ STRUCTURED AI ANALYSIS ============
def get_structured_report(resume: str, job_desc: str) -> ResumeAnalysisOutput:
    """
    Use LangChain with structured output to get consistent, parseable analysis
    """
    if not api_key:
        st.error("API Key not loaded. Cannot generate report.")
        raise ValueError("API Key not found.")

    system_prompt = """You are an expert AI Resume Analyzer specializing in matching candidates to job requirements.

Your task is to analyze the resume against the job description and provide a structured evaluation.

CRITICAL INSTRUCTIONS:
1. Dynamically identify 5-7 KEY CRITERIA that are most relevant to THIS specific job description
2. DO NOT use generic criteria - analyze what THIS job actually requires
3. For each criterion, provide:
   - A clear, specific criterion name
   - A score from 1.0 to 5.0 (you can use decimals like 3.5)
   - A detailed explanation with evidence from the resume
4. Be objective, evidence-based, and constructive
5. Focus on job requirements like: relevant experience, required skills, certifications, education, location match, availability, soft skills, etc.

Provide comprehensive pros and cons lists, and an overall assessment with actionable improvement suggestions."""

    try:
        # Initialize ChatGroq with LangChain
        llm = ChatGroq(
            api_key=api_key,
            model=MODEL_ID,
            temperature=0.2
        )
        
        # Bind structured output schema to the model - LangChain's way
        structured_llm = llm.with_structured_output(
            ResumeAnalysisOutput,
            method="function_calling",  # Uses Groq's tool-calling API
            include_raw=False
        )
        
        # Create messages using LangChain message types
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"""Analyze the following resume against the job description:

JOB DESCRIPTION:
{job_desc}

RESUME:
{resume}

Provide a comprehensive structured analysis."""
            )
        ]
        
        # Invoke the model - LangChain's invoke() method
        st.info("🔄 Sending resume to AI for structured analysis...")
        start_time = time.time()
        analysis = structured_llm.invoke(messages)
        duration = time.time() - start_time
        st.success(f"✅ Analysis completed in {duration:.2f} seconds")
        
        return analysis
        
    except Exception as e:
        st.error(f"AI model error: {str(e)}")
        raise

def compute_ai_score_from_structured(analysis: ResumeAnalysisOutput) -> float:
    """Calculate average score from structured output"""
    if not analysis.criteria_scores:
        return 0.0
    
    total = sum(criterion.score for criterion in analysis.criteria_scores)
    avg_5pt = total / len(analysis.criteria_scores)
    return avg_5pt / 5.0

def convert_structured_to_markdown(analysis: ResumeAnalysisOutput) -> str:
    """Convert structured output to markdown for display and PDF"""
    md_parts = []
    
    # Criteria sections
    for criterion in analysis.criteria_scores:
        md_parts.append(f"## {criterion.criterion_name}")
        md_parts.append(f"**Score: {criterion.score}/5**\n")
        md_parts.append(f"{criterion.explanation}\n")
    
    # Pros
    md_parts.append("## Pros of the Resume (Based on Job Description)")
    for pro in analysis.pros:
        md_parts.append(f"- {pro}")
    md_parts.append("")
    
    # Cons
    md_parts.append("## Cons of the Resume (Based on Job Description)")
    for con in analysis.cons:
        md_parts.append(f"- {con}")
    md_parts.append("")
    
    # Overall assessment
    md_parts.append("## Overall Match Assessment")
    md_parts.append(analysis.overall_assessment)
    md_parts.append("")
    
    # Requirements met
    if analysis.key_requirements_met:
        md_parts.append("## Key Requirements Met")
        for req in analysis.key_requirements_met:
            md_parts.append(f"- {req}")
        md_parts.append("")
    
    # Requirements missing
    if analysis.key_requirements_missing:
        md_parts.append("## Key Requirements Missing")
        for req in analysis.key_requirements_missing:
            md_parts.append(f"- {req}")
        md_parts.append("")
    
    return "\n".join(md_parts)

# ============ PDF BUILDER ============
def build_pdf_bytes(*, logo_path: str | None, candidate_name: str, ats_score: float, ai_score: float, report_markdown: str) -> bytes:
    """
    Clean, single-column PDF with dynamic sections
    """
    from io import BytesIO
    buff = BytesIO()

    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=45, rightMargin=45, topMargin=45, bottomMargin=45)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=22, alignment=1, textColor=colors.HexColor("#0B5345"))
    subtitle = ParagraphStyle("subtitle", parent=styles["Heading2"], fontSize=12, alignment=1, textColor=colors.HexColor("#34495E"))
    section = ParagraphStyle("section", parent=styles["Heading2"], textColor=colors.HexColor("#154360"))
    kpi = ParagraphStyle("kpi", parent=styles["Heading3"], textColor=colors.HexColor("#1A5276"))
    body = ParagraphStyle("body", parent=styles["BodyText"], leading=15, spaceAfter=6)
    candidate_style = ParagraphStyle("candidate", parent=styles["Heading2"], alignment=1, textColor=colors.HexColor("#1F618D"))

    story = []

    if logo_path and Path(logo_path).exists():
        story.append(Image(logo_path, width=70, height=70))
        story.append(Spacer(1, 6))

    story.append(Paragraph("Aspect AI Resume Analyzer", title))
    story.append(Paragraph("Advanced AI-Powered Resume Analysis and Job Matching", subtitle))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(color=colors.HexColor("#D5D8DC"), thickness=1))
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"Candidate: <b>{candidate_name or '—'}</b>", candidate_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Understanding the Scores", section))
    story.append(Paragraph("Each criterion below is rated from 1 to 5 based on job fit and supporting evidence.", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph(f"ATS Similarity Score: <b>{ats_score*100:.1f}%</b>", kpi))
    story.append(Paragraph(f"AI Evaluation Score: <b>{ai_score*100:.1f}%</b>", kpi))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(color=colors.HexColor("#D5D8DC"), thickness=1))
    story.append(Spacer(1, 8))

    for raw in report_markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("## "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(line.replace("##", "").strip(), section))
            continue
        if line.startswith(("-", "*", "•")):
            line = line.lstrip("-*• ").strip()
        story.append(Paragraph(line, body))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(color=colors.HexColor("#D5D8DC"), thickness=1))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Generated by Aspect AI Resume Analyzer — empowering data-driven hiring decisions.", body))

    doc.build(story)
    buff.seek(0)
    return buff.read()

# ============ SESSION ============
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""
if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = ""

# ============ HERO ============
logo_b64 = get_image_as_base64(LOGO_FILE)
if logo_b64:
    st.markdown(f"""
    <div class="hero-header">
        <img src="data:image/png;base64,{logo_b64}" alt="Aspect AI Logo">
        <h1>Aspect AI Resume Analyzer</h1>
        <p>Advanced AI-powered resume analysis and job matching</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero-header">
        <h1>Aspect AI Resume Analyzer</h1>
        <p>Advanced AI-powered resume analysis and job matching</p>
    </div>
    """, unsafe_allow_html=True)

# ============ UI / FLOW ============
if not api_key:
    st.error("❌ GROQ_API_KEY not found.")
    st.stop()

if not st.session_state.form_submitted:
    with st.form("my_form"):
        st.markdown("### 👤 Candidate Name (for report header)")
        name_input = st.text_input("Full name", placeholder="e.g., John Smith")

        st.markdown("### 📄 Upload Candidate Resume")
        resume_file = st.file_uploader(label="Upload Resume/CV (PDF)", type="pdf")

        st.markdown("### 💼 Job Description")
        job_desc_input = st.text_area("Paste the full job description here:", height=220)

        submitted = st.form_submit_button("🚀 Analyze Resume", use_container_width=True)

        if submitted:
            if job_desc_input and resume_file:
                with st.spinner("🔍 Extracting resume text..."):
                    text = extract_pdf_text(resume_file)
                if text:
                    st.session_state.job_desc = job_desc_input
                    st.session_state.resume = text
                    st.session_state.candidate_name = (name_input or "").strip()
                    st.session_state.form_submitted = True
                    st.rerun()
                else:
                    st.error("❌ Failed to read the PDF. Please try another file.")
            else:
                st.warning("⚠️ Please upload a resume and provide a job description.")
else:
    with st.spinner("⚡ Generating structured AI analysis..."):
        # Calculate ATS score
        ats_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)
        
        # Get structured analysis from AI
        try:
            structured_analysis = get_structured_report(st.session_state.resume, st.session_state.job_desc)
            ai_score = compute_ai_score_from_structured(structured_analysis)
            report_md = convert_structured_to_markdown(structured_analysis)
        except Exception as e:
            st.error(f"Failed to generate analysis: {str(e)}")
            st.stop()

    # KPI cards
    st.markdown('<div class="score-container">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="score-card"><h3>📊 ATS Similarity Score</h3>'
            f'<h2>{ats_score:.1%}</h2>'
            f'<p style="margin-top: 1rem; opacity: 0.9;">Resume–Job Description Match</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="score-card"><h3>🎯 AI Evaluation Score</h3>'
            f'<h2>{ai_score:.1%}</h2>'
            f'<p style="margin-top: 1rem; opacity: 0.9;">Average across evaluation criteria</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📌 What These Scores Mean")
    st.markdown("Each criterion below is rated from **1 to 5** based on job fit and supporting evidence from the resume.")

    # Full report
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(f"## Candidate: **{st.session_state.candidate_name or '—'}**")
    st.markdown("## AI Generated Analysis Report")
    st.markdown(report_md, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # PDF download
    pdf_bytes = build_pdf_bytes(
        logo_path=LOGO_FILE if Path(LOGO_FILE).exists() else None,
        candidate_name=st.session_state.candidate_name or "—",
        ats_score=ats_score,
        ai_score=ai_score,
        report_markdown=report_md
    )
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"Aspect_AI_Resume_Report_{(st.session_state.candidate_name or 'Candidate').replace(' ','_')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    if st.button(" Analyze Another", use_container_width=True, type="primary"):
        st.session_state.form_submitted = False
        st.session_state.resume = ""
        st.session_state.job_desc = ""
        st.session_state.candidate_name = ""
        st.rerun()
