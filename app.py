import os
import io
import html
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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# MINIMAL EDITORIAL STYLING
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

    /* Tighten default Streamlit top spacing */
    .block-container {
        padding-top: 2.25rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Remove anchor link icons next to headings */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }

    /* Subtle editorial page header */
    .page-header {
        border-bottom: 1px solid rgba(148, 163, 184, 0.28);
        padding-bottom: 1.15rem;
        margin-bottom: 1.75rem;
    }
    .page-kicker {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .page-title {
        font-size: 1.95rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin: 0 0 0.45rem 0;
    }
    .page-subtitle {
        font-size: 0.96rem;
        opacity: 0.78;
        margin: 0;
        max-width: 780px;
        line-height: 1.5;
    }

    /* Section labels */
    .section-label {
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.65rem;
    }

    /* Staged file rows */
    .file-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0.65rem;
        margin-top: 0.35rem;
        border-radius: 6px;
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.22);
        font-size: 0.82rem;
    }
    .file-size {
        color: #64748B;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
    }

    /* Workflow steps list */
    .step-item {
        font-size: 0.83rem;
        line-height: 1.45;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
    }
    .step-item:last-child {
        border-bottom: none;
    }
    .step-num {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: #1D4ED8;
        margin-right: 0.4rem;
    }

    /* Clean primary button override (replaces default bright red) */
    button[kind="primary"] {
        background-color: #1E3A8A !important;
        border-color: #1E3A8A !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.55rem 1rem !important;
    }
    button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }
    button[kind="secondary"] {
        border-radius: 6px !important;
        font-weight: 500 !important;
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
<div class="page-header">
    <div class="page-kicker">Literature Synthesis Workspace</div>
    <h1 class="page-title">Autonomous Literature Review Assistance Agent</h1>
    <p class="page-subtitle">
        Cross-reference uploaded research manuscripts with web search and automated verification to compile structured, citation-backed literature reviews.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TWO-COLUMN WORKSPACE
# --------------------------------------------------

left_col, right_col = st.columns([1, 1.55], gap="large")

with left_col:

    with st.container(border=True):
        st.markdown('<div class="section-label">Research Configuration</div>', unsafe_allow_html=True)

        topic = st.text_input(
            "Research Topic",
            placeholder="e.g., Generative AI applications in clinical diagnostics"
        )

        uploaded_files = st.file_uploader(
            "Source Manuscripts (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF papers to index into the vector store."
        )

        if uploaded_files:
            for f in uploaded_files:
                size_kb = len(f.getbuffer()) / 1024
                size_str = f"{size_kb / 1024:.2f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
                st.markdown(
                    f'<div class="file-row"><span>{html.escape(f.name)}</span>'
                    f'<span class="file-size">{size_str}</span></div>',
                    unsafe_allow_html=True
                )
            st.write("")

        run_clicked = st.button(
            "Run Literature Review",
            type="primary",
            use_container_width=True
        )

    with st.container(border=True):
        st.markdown('<div class="section-label">Pipeline Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="step-item"><span class="step-num">01</span><strong>Document Indexing</strong> — Splits uploaded PDFs and stores embeddings in ChromaDB.</div>
        <div class="step-item"><span class="step-num">02</span><strong>RAG &amp; Web Retrieval</strong> — Extracts manuscript evidence and live DuckDuckGo results.</div>
        <div class="step-item"><span class="step-num">03</span><strong>Verification</strong> — Cross-checks document claims against external findings.</div>
        <div class="step-item"><span class="step-num">04</span><strong>Synthesis</strong> — Compiles an academic report with Markdown and PDF exports.</div>
        """, unsafe_allow_html=True)


with right_col:

    if run_clicked:

        if not topic.strip():

            st.error("Please enter a research topic before running the review.")

        elif not uploaded_files:

            st.error("Please upload at least one PDF manuscript.")

        else:

            try:

                with st.status("Executing literature review pipeline...", expanded=True) as status:

                    st.write("Saving uploaded PDF manuscripts...")
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(
                            DATA_DIR,
                            os.path.basename(uploaded_file.name)
                        )
                        with open(file_path, "wb") as file:
                            file.write(uploaded_file.getbuffer())

                    st.write("Chunking documents and updating Chroma vector store...")
                    ingest_pdfs()

                    st.write("Running retrieval, web search, verification, and synthesis agents...")
                    result = research_workflow.invoke({
                        "topic": topic.strip(),
                        "rag_result": "",
                        "web_result": "",
                        "verification": "",
                        "report": ""
                    })

                    report = result["report"]

                    st.write("Formatting Markdown and PDF deliverables...")
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

                    status.update(label="Literature review completed.", state="complete", expanded=False)

            except Exception as error:

                err_msg = str(error)
                if "ollama" in err_msg.lower() or "connection" in err_msg.lower():
                    st.error(
                        "Unable to reach the local Ollama inference service. "
                        "Ensure Ollama is running locally (`ollama serve`) with `llama3.2` and `nomic-embed-text` pulled."
                    )
                else:
                    st.error("The literature review workflow encountered an execution error.")

                with st.expander("Technical error details"):
                    st.code(err_msg)

    # --------------------------------------------------
    # DISPLAY REPORT AND DOWNLOAD OPTIONS
    # --------------------------------------------------

    if "research_report" in st.session_state:

        with st.container(border=True):
            st.subheader("Final Research Report")
            st.caption("Topic: " + st.session_state["research_topic"])

            tab_report, tab_evidence, tab_exports = st.tabs([
                "Report",
                "Evidence & Verification",
                "Exports"
            ])

            with tab_report:
                st.markdown(st.session_state["research_report"])

            with tab_evidence:
                if st.session_state.get("research_verification"):
                    st.markdown("#### Cross-Source Verification")
                    st.markdown(st.session_state["research_verification"])
                    st.divider()

                if st.session_state.get("research_rag"):
                    st.markdown("#### Document Retrieval (RAG) Findings")
                    st.markdown(st.session_state["research_rag"])
                    st.divider()

                if st.session_state.get("research_web"):
                    st.markdown("#### Web Search Findings")
                    st.markdown(st.session_state["research_web"])

            with tab_exports:
                st.write("Download the compiled review or inspect the saved local copies in `outputs/`.")
                dl_col1, dl_col2 = st.columns(2)

                with dl_col1:
                    st.download_button(
                        label="Download Markdown (.md)",
                        data=st.session_state["research_report"],
                        file_name="research_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

                with dl_col2:
                    st.download_button(
                        label="Download PDF (.pdf)",
                        data=st.session_state["research_pdf"],
                        file_name="research_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                st.caption("Saved locally to `outputs/research_report.md` and `outputs/research_report.pdf`.")

    elif not run_clicked:

        with st.container(border=True):
            st.markdown('<div class="section-label">Report Output</div>', unsafe_allow_html=True)
            st.markdown(
                "Enter a research topic and upload one or more PDF papers on the left, then click **Run Literature Review**. "
                "The synthesized report, verification findings, and downloadable Markdown and PDF files will appear in this panel."
            )
