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

# ---------- PDF (ReportLab) ----------
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable
)

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
LOGO_FILE = str(SCRIPT_DIR / "images.png")  # keep your logo here

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

# --------- Stable, TITLED sections parser (fixes score swing) ---------
TITLED_SECTIONS = [
    "1. Experience as a Qualified Electrician",
    "2. Certifications (BS 7671, 2391)",
    "3. Self-Employment / Apprenticeship Readiness",
    "4. UK Manual Driving Licence",
    "5. Availability (40 Hours/Week)",
    "6. Knowledge of Regulations & Best Practices",
    "7. Communication & Customer Service",
]

def split_markdown_sections(md: str):
    sections = {}
    current_title = None
    buf = []
    for line in md.splitlines():
        if line.strip().startswith("## "):
            if current_title is not None:
                sections[current_title] = "\n".join(buf).strip()
            current_title = line.strip().replace("##", "", 1).strip()
            buf = []
        else:
            buf.append(line)
    if current_title is not None:
        sections[current_title] = "\n".join(buf).strip()
    return sections

def extract_first_score(text):
    m = re.search(r'(\d+(?:\.\d+)?)/5', text)
    return float(m.group(1)) if m else None

def compute_ai_eval_from_titled_sections(report_md: str):
    sections = split_markdown_sections(report_md)
    scores = []
    for title in TITLED_SECTIONS:
        matched_key = next((k for k in sections.keys() if k.lower().startswith(title.lower())), None)
        if matched_key:
            s = extract_first_score(sections[matched_key])
            if s is not None:
                scores.append(s)
    if not scores:
        return 0.0
    avg_5pt = sum(scores) / len(scores)
    return avg_5pt / 5.0

# ============ AI PROMPT ============
def get_report(resume, job_desc):
    if not api_key:
        st.error("API Key not loaded. Cannot generate report.")
        return "Error: API Key not found."

    # Enforce EXACT titled sections; start with "N/5" but NO emojis (keeps PDF clean)
    prompt = f"""
You are an expert AI Resume Analyzer.

Return the result in MARKDOWN with EXACTLY these sections (start each section title with '## '):

## 1. Experience as a Qualified Electrician
- Begin with a score like "4/5" then a short paragraph.

## 2. Certifications (BS 7671, 2391)
- Score "N/5" + explanation.

## 3. Self-Employment / Apprenticeship Readiness
- Score "N/5" + explanation.

## 4. UK Manual Driving Licence
- Score "N/5" + explanation.

## 5. Availability (40 Hours/Week)
- Score "N/5" + explanation.

## 6. Knowledge of Regulations & Best Practices
- Score "N/5" + explanation.

## 7. Communication & Customer Service
- Score "N/5" + explanation.

## Pros of the Resume (Based on Job Description)
- Bullet list of strengths.

## Cons of the Resume (Based on Job Description)
- Bullet list of gaps.

## Why this is Good/Bad
- One concise paragraph with improvements.

Return only markdown content.

Resume:
```{resume}```

Job Description:
```{job_desc}```
"""
    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_ID,
            temperature=0.0,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error("AI model error:")
        st.error(e)
        return "Failed to generate report due to an API error."

# ============ PDF BUILDER ============
def build_pdf_bytes(*, logo_path: str | None, candidate_name: str, ats_score: float, ai_score: float, report_markdown: str) -> bytes:
    """
    Clean, single-column PDF:
    - Logo, title, candidate name
    - One-line 'What These Scores Mean'
    - ATS & AI KPI values
    - Titled sections rendered as paragraphs (no bullets/boxes/emojis)
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

    # ONE-LINE explanation only (per your request)
    story.append(Paragraph("Understanding the Scores", section))
    story.append(Paragraph("Each criterion below is rated from 1 to 5 based on job fit and supporting evidence.", body))
    story.append(Spacer(1, 8))

    # KPIs
    story.append(Paragraph(f"ATS Similarity Score: <b>{ats_score*100:.1f}%</b>", kpi))
    story.append(Paragraph(f"AI Evaluation Score: <b>{ai_score*100:.1f}%</b>", kpi))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(color=colors.HexColor("#D5D8DC"), thickness=1))
    story.append(Spacer(1, 8))

    # Render titled sections: headers + paragraphs ONLY (no bullets)
    for raw in report_markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("## "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(line.replace("##", "").strip(), section))
            continue
        # Convert bullet lines to plain paragraphs (remove symbols)
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
        name_input = st.text_input("Full name", placeholder="e.g., Numan Ermis")

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
    with st.spinner("⚡ Generating AI analysis..."):
        ats_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)
        report_md = get_report(st.session_state.resume, st.session_state.job_desc)
        avg_score = compute_ai_eval_from_titled_sections(report_md)  # stable

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
            f'<h2>{avg_score:.1%}</h2>'
            f'<p style="margin-top: 1rem; opacity: 0.9;">Average across titled criteria</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # What These Scores Mean (ONE LINE ONLY)
    st.markdown("### 📌 What These Scores Mean")
    st.markdown("Each criterion below is rated from **1 to 5** based on job fit and supporting evidence.")

    # Full titled report
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(f"## Candidate: **{st.session_state.candidate_name or '—'}**")
    st.markdown("## AI Generated Analysis Report (Titled Sections)")
    st.markdown(report_md, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # PDF download (clean, branded, no bullets/boxes)
    pdf_bytes = build_pdf_bytes(
        logo_path=LOGO_FILE if Path(LOGO_FILE).exists() else None,
        candidate_name=st.session_state.candidate_name or "—",
        ats_score=ats_score,
        ai_score=avg_score,
        report_markdown=report_md
    )
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"Aspect_AI_Resume_Report_{(st.session_state.candidate_name or 'Candidate').replace(' ','_')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    if st.button("🔙 Analyze Another", use_container_width=True, type="primary"):
        st.session_state.form_submitted = False
        st.session_state.resume = ""
        st.session_state.job_desc = ""
        st.session_state.candidate_name = ""
        st.rerun()
