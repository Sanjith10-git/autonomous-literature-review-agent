import os
import io
import html
import tempfile
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable
)

from config.llm import get_groq_api_key, is_ollama_running
from rag.ingest import ingest_pdfs
from workflow import research_workflow


# --------------------------------------------------
# PROJECT FOLDERS
# --------------------------------------------------

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Autonomous Literature Review Assistance Agent",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# CLEAN EDITORIAL DESIGN SYSTEM
# --------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    code, pre {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Balanced centered container */
    .block-container {
        padding-top: 2.75rem;
        padding-bottom: 4rem;
        max-width: 860px;
    }

    /* Hide distracting Streamlit anchor links */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }

    /* Editorial Header */
    .hero-wrap {
        padding: 0 0 1.35rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.28);
    }
    .hero-accent {
        width: 44px;
        height: 4px;
        background-color: #1E3A8A;
        border-radius: 2px;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1.18;
        margin: 0 0 0.5rem 0;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        opacity: 0.76;
        line-height: 1.55;
        margin: 0;
    }

    /* Elevated native containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        box-shadow: 0 6px 24px -6px rgba(15, 23, 42, 0.07) !important;
        padding: 0.4rem 0.25rem !important;
    }

    /* Input focus polish */
    [data-testid="stTextInput"] input {
        border-radius: 8px !important;
        padding: 0.6rem 0.8rem !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #1D4ED8 !important;
        box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.14) !important;
    }

    /* File uploader subtle styling */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 10px !important;
        border: 1.5px dashed rgba(100, 116, 139, 0.4) !important;
        transition: border-color 0.2s ease, background-color 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #1D4ED8 !important;
    }

    /* Staged manuscript pills */
    .staged-files {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.35rem 0 0.85rem 0;
    }
    .staged-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.32rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .staged-size {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        color: #64748B;
    }

    /* Primary & secondary buttons */
    button[kind="primary"] {
        background-color: #1E3A8A !important;
        border-color: #1E3A8A !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        padding: 0.62rem 1.2rem !important;
        box-shadow: 0 2px 6px rgba(30, 58, 138, 0.2) !important;
        transition: background-color 0.18s ease, transform 0.15s ease, box-shadow 0.18s ease !important;
    }
    button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.28) !important;
    }
    button[kind="secondary"] {
        border-radius: 8px !important;
        font-weight: 500 !important;
        border-color: rgba(148, 163, 184, 0.38) !important;
    }
    button[kind="secondary"]:hover {
        border-color: #1D4ED8 !important;
        color: #1D4ED8 !important;
    }

    /* Clean tab highlight */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1D4ED8 !important;
    }
    [data-baseweb="tab-highlight"] {
        background-color: #1D4ED8 !important;
    }

    /* Report reading typography */
    .report-body {
        line-height: 1.7;
        font-size: 0.96rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PDF GENERATION FUNCTION
# --------------------------------------------------

def create_pdf(report, topic):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1E3A8A"),
        fontSize=18,
        leading=22,
        spaceAfter=10
    )

    topic_style = ParagraphStyle(
        "TopicHeader",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#334155"),
        fontSize=11,
        leading=15,
        spaceAfter=12
    )

    heading_style = styles["Heading2"]
    subheading_style = styles["Heading3"]
    body_style = styles["BodyText"]

    story = []

    story.append(
        Paragraph(
            "Autonomous Literature Review Assistance Agent",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Research Topic: " + html.escape(topic),
            topic_style
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.75,
            color=colors.HexColor("#CBD5E1"),
            spaceAfter=14
        )
    )

    for line in report.splitlines():

        line = line.strip()

        if not line:
            story.append(Spacer(1, 6))
            continue

        if line.startswith("# "):

            story.append(
                Paragraph(
                    html.escape(line[2:]),
                    heading_style
                )
            )

        elif line.startswith("## "):

            story.append(
                Paragraph(
                    html.escape(line[3:]),
                    heading_style
                )
            )

        elif line.startswith("### "):

            story.append(
                Paragraph(
                    html.escape(line[4:]),
                    subheading_style
                )
            )

        elif line.startswith("- ") or line.startswith("* "):

            story.append(
                Paragraph(
                    "&bull; " + html.escape(line[2:]),
                    body_style
                )
            )
            story.append(Spacer(1, 3))

        else:

            # Display ordinary text safely in the PDF.
            story.append(
                Paragraph(
                    html.escape(line),
                    body_style
                )
            )

            story.append(Spacer(1, 5))

    document.build(story)

    return buffer.getvalue()


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero-wrap">
    <div class="hero-accent"></div>
    <h1 class="hero-title">Autonomous Literature Review Assistance Agent</h1>
    <p class="hero-subtitle">
        Upload peer-reviewed manuscripts and specify a research topic to generate a verified, citation-backed synthesis report.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# RESEARCH INPUT CARD
# --------------------------------------------------

configured_groq_key = get_groq_api_key()
local_ollama_up = is_ollama_running()
groq_key_input = configured_groq_key

with st.container(border=True):
    topic = st.text_input(
        "Research Topic",
        placeholder="Example: Generative AI applications in clinical diagnostics"
    )

    uploaded_files = st.file_uploader(
        "Research Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        chips_html = "".join(
            f'<span class="staged-chip">{html.escape(f.name)}'
            f'<span class="staged-size">{len(f.getbuffer()) / 1024:.1f} KB</span></span>'
            for f in uploaded_files
        )
        st.markdown(f'<div class="staged-files">{chips_html}</div>', unsafe_allow_html=True)

    if not local_ollama_up and not configured_groq_key:
        groq_key_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Required on Streamlit Cloud when GROQ_API_KEY is not saved in App Secrets."
        )

    run_clicked = st.button(
        "Start Research",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# WORKFLOW EXECUTION
# --------------------------------------------------

if run_clicked:

    if not topic.strip():

        st.error("Please enter a research topic.")

    elif not uploaded_files:

        st.error("Please upload at least one PDF research paper.")

    elif not local_ollama_up and not groq_key_input.strip():

        st.error(
            "Please enter a Groq API Key above (or add `GROQ_API_KEY` in Streamlit Cloud App Settings -> Secrets)."
        )

    else:

        try:

            with st.status("Running literature review...", expanded=True) as status:

                st.write("Saving uploaded PDF files...")
                temp_dir = tempfile.mkdtemp()
                uploaded_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(
                        temp_dir,
                        os.path.basename(uploaded_file.name)
                    )
                    with open(file_path, "wb") as file:
                        file.write(uploaded_file.getbuffer())
                    uploaded_paths.append(file_path)

                st.write("Indexing manuscripts into vector store...")
                ingest_pdfs(pdf_paths=uploaded_paths)

                st.write("Retrieving document evidence, searching web literature, and verifying claims...")
                result = research_workflow.invoke({
                    "topic": topic.strip(),
                    "rag_result": "",
                    "web_result": "",
                    "verification": "",
                    "report": "",
                    "api_key": groq_key_input.strip() or None
                })

                report = result["report"]

                st.write("Generating Markdown and PDF copies...")
                markdown_path = os.path.join(
                    OUTPUT_DIR,
                    "research_report.md"
                )
                with open(markdown_path, "w", encoding="utf-8") as file:
                    file.write(report)

                pdf_bytes = create_pdf(
                    report,
                    topic.strip()
                )

                pdf_path = os.path.join(
                    OUTPUT_DIR,
                    "research_report.pdf"
                )
                with open(pdf_path, "wb") as file:
                    file.write(pdf_bytes)

                # Keep results available in the current session.
                st.session_state["research_report"] = report
                st.session_state["research_pdf"] = pdf_bytes
                st.session_state["research_topic"] = topic.strip()
                st.session_state["research_rag"] = result.get("rag_result", "")
                st.session_state["research_web"] = result.get("web_result", "")
                st.session_state["research_verification"] = result.get("verification", "")

                status.update(label="Research completed.", state="complete", expanded=False)

        except Exception as error:

            err_msg = str(error)
            if "no active llm backend" in err_msg.lower() or "ollama" in err_msg.lower() or "connection" in err_msg.lower():
                st.error(
                    "Unable to reach an active LLM backend. On Streamlit Cloud, add `GROQ_API_KEY` in App Settings -> Secrets "
                    "(or enter your key above). Locally, start Ollama (`ollama serve`)."
                )
            else:
                st.error("The research workflow failed during execution.")

            with st.expander("Technical error details"):
                st.code(err_msg)


# --------------------------------------------------
# GENERATED REPORT (SHOWN ONLY AFTER EXECUTION)
# --------------------------------------------------

if "research_report" in st.session_state:

    st.write("")
    with st.container(border=True):

        header_left, header_right = st.columns([2.2, 1.3])
        with header_left:
            st.subheader("Research Report")
            st.caption("Topic: " + st.session_state["research_topic"])
        with header_right:
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="Markdown",
                    data=st.session_state["research_report"],
                    file_name="research_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with dl_col2:
                st.download_button(
                    label="PDF",
                    data=st.session_state["research_pdf"],
                    file_name="research_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        tab_report, tab_verification = st.tabs([
            "Report",
            "Verification & Sources"
        ])

        with tab_report:
            st.markdown('<div class="report-body">', unsafe_allow_html=True)
            st.markdown(st.session_state["research_report"])
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_verification:
            if st.session_state.get("research_verification"):
                st.markdown("#### Cross-Source Verification")
                st.markdown(st.session_state["research_verification"])
                st.divider()

            if st.session_state.get("research_rag"):
                st.markdown("#### Uploaded Document Findings")
                st.markdown(st.session_state["research_rag"])
                st.divider()

            if st.session_state.get("research_web"):
                st.markdown("#### Web Search Findings")
                st.markdown(st.session_state["research_web"])
