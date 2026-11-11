import os
import io
import html
import json
import uuid
import math
import tempfile
from datetime import datetime
import streamlit as st
from pypdf import PdfReader

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

from config.llm import is_ollama_running
from rag.ingest import ingest_pdfs
from agents.rag_agent import rag_agent
from workflow import research_workflow


# --------------------------------------------------
# PROJECT FOLDERS & SESSION STATE
# --------------------------------------------------

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(
    page_title="Autonomous Literature Review Assistance Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "history" not in st.session_state:
    st.session_state["history"] = []
if "active_review_id" not in st.session_state:
    st.session_state["active_review_id"] = None
if "topic_input" not in st.session_state:
    st.session_state["topic_input"] = ""
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0


# --------------------------------------------------
# BESPOKE EDITORIAL & ACADEMIC DESIGN SYSTEM
# --------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,600;6..72,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    code, pre {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Main canvas proportions */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4.5rem;
        max-width: 960px;
    }

    /* Hide anchor links next to headings */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }

    /* Sidebar visual styling */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.22);
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.2rem 0 0.9rem 0;
        margin-bottom: 0.9rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.22);
    }
    .sidebar-brand-icon {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: linear-gradient(180deg, #B91C1C 0%, #7F1D1D 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(153, 27, 27, 0.28);
    }
    .sidebar-brand-title {
        font-size: 0.96rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        line-height: 1.2;
    }
    .sidebar-section-label {
        font-size: 0.71rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B;
        margin: 0.45rem 0 0.35rem 0;
    }

    /* Left-aligned clean sidebar history buttons */
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        justify-content: flex-start !important;
        text-align: left !important;
        border-radius: 8px !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        padding: 0.48rem 0.7rem !important;
        transition: all 0.16s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button p {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        width: 100%;
    }

    /* Subtle hover-revealed popover buttons */
    [data-testid="stPopover"] > button {
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        opacity: 0.85;
        transition: all 0.16s ease !important;
    }
    [data-testid="stPopover"] > button:hover {
        opacity: 1;
        border-color: #B91C1C !important;
        color: #B91C1C !important;
    }

    /* Professional Editorial Header Card */
    .hero-card {
        display: flex;
        align-items: center;
        gap: 1.15rem;
        padding: 1.35rem 1.7rem;
        margin-bottom: 1.35rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.08) 0%, rgba(185, 28, 28, 0.025) 55%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(153, 27, 27, 0.2);
        border-left: 5px solid #991B1B;
        box-shadow: 0 10px 28px -12px rgba(127, 29, 29, 0.18);
    }
    .hero-icon {
        width: 46px;
        height: 46px;
        border-radius: 11px;
        background: linear-gradient(180deg, #B91C1C 0%, #7F1D1D 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 4px 14px rgba(153, 27, 27, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.22);
    }
    .hero-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.05rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1.15;
        margin: 0;
    }

    /* Elevated paper-like containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        box-shadow: 0 10px 28px -10px rgba(15, 23, 42, 0.08) !important;
        padding: 0.5rem 0.35rem !important;
    }

    /* Clean text input & dropzone */
    [data-testid="stTextInput"] input {
        border-radius: 9px !important;
        padding: 0.6rem 0.85rem !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #B91C1C !important;
        box-shadow: 0 0 0 3px rgba(185, 28, 28, 0.14) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 10px !important;
        border: 1.5px dashed rgba(100, 116, 139, 0.38) !important;
        transition: border-color 0.2s ease, background-color 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #B91C1C !important;
        background-color: rgba(185, 28, 28, 0.025) !important;
    }

    /* Staged manuscript badges */
    .staged-files {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.4rem 0 0.9rem 0;
    }
    .staged-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.36rem 0.72rem;
        border-radius: 7px;
        font-size: 0.8rem;
        font-weight: 500;
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.26);
    }
    .staged-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.73rem;
        color: #64748B;
    }

    /* Primary & secondary buttons */
    button[kind="primary"] {
        background: linear-gradient(180deg, #B91C1C 0%, #991B1B 100%) !important;
        border-color: #7F1D1D !important;
        color: #FFFFFF !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.62rem 1.25rem !important;
        box-shadow: 0 3px 10px rgba(153, 27, 27, 0.24) !important;
        transition: all 0.18s ease !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(180deg, #DC2626 0%, #B91C1C 100%) !important;
        border-color: #991B1B !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(185, 28, 28, 0.32) !important;
    }
    button[kind="secondary"] {
        border-radius: 8px !important;
        font-weight: 500 !important;
        border-color: rgba(148, 163, 184, 0.35) !important;
    }
    button[kind="secondary"]:hover {
        border-color: #B91C1C !important;
        color: #B91C1C !important;
    }

    /* Segmented Pill Tab Styling */
    [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(148, 163, 184, 0.1);
        padding: 0.3rem;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    button[data-baseweb="tab"] {
        border-radius: 7px !important;
        padding: 0.42rem 0.9rem !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(153, 27, 27, 0.12) !important;
        color: #B91C1C !important;
        font-weight: 600 !important;
    }
    [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Micro Metric Strip */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.65rem;
        margin: 0.4rem 0 1.15rem 0;
    }
    .metric-box {
        padding: 0.65rem 0.85rem;
        border-radius: 9px;
        background: rgba(148, 163, 184, 0.07);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.92rem;
        font-weight: 600;
    }

    /* Reading flow inside Report */
    .report-body {
        line-height: 1.75;
        font-size: 0.96rem;
        padding-top: 0.35rem;
    }
    .report-body blockquote {
        border-left: 3px solid #B91C1C;
        padding-left: 1rem;
        opacity: 0.88;
    }
    .qa-answer {
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        background: rgba(153, 27, 27, 0.05);
        border: 1px solid rgba(153, 27, 27, 0.18);
        margin-top: 0.5rem;
        line-height: 1.65;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def inspect_pdf_metadata(uploaded_file) -> dict:
    """Extracts page count and size for staged PDF files."""
    raw_bytes = uploaded_file.getbuffer()
    size_kb = len(raw_bytes) / 1024
    pages = "?"
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = len(reader.pages)
    except Exception:
        pass
    return {
        "name": uploaded_file.name,
        "pages": pages,
        "size_str": f"{size_kb / 1024:.2f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
    }


def create_pdf(report: str, topic: str) -> bytes:
    """Generates a styled A4 PDF copy of the literature review report."""
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
        textColor=colors.HexColor("#991B1B"),
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

    story = [
        Paragraph("Autonomous Literature Review Assistance Agent", title_style),
        Paragraph("Research Topic: " + html.escape(topic), topic_style),
        HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#CBD5E1"), spaceAfter=14)
    ]

    for line in report.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue

        if line.startswith("# "):
            story.append(Paragraph(html.escape(line[2:]), heading_style))
        elif line.startswith("## "):
            story.append(Paragraph(html.escape(line[3:]), heading_style))
        elif line.startswith("### "):
            story.append(Paragraph(html.escape(line[4:]), subheading_style))
        elif line.startswith("- ") or line.startswith("* "):
            story.append(Paragraph("&bull; " + html.escape(line[2:]), body_style))
            story.append(Spacer(1, 3))
        else:
            story.append(Paragraph(html.escape(line), body_style))
            story.append(Spacer(1, 5))

    document.build(story)
    return buffer.getvalue()


def get_active_review():
    active_id = st.session_state.get("active_review_id")
    for item in st.session_state.get("history", []):
        if item["id"] == active_id:
            return item
    return None


def start_new_review():
    st.session_state["active_review_id"] = None
    st.session_state["topic_input"] = ""
    st.session_state["uploader_key"] += 1


# --------------------------------------------------
# SIDEBAR: HISTORY, RESEARCH PREFERENCES & NOTES
# --------------------------------------------------

local_ollama_up = is_ollama_running()

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>
                <path d="M8 7h6"/>
                <path d="M8 11h8"/>
            </svg>
        </div>
        <div>
            <div class="sidebar-brand-title">Literature Agent</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("+ New Review", use_container_width=True):
        start_new_review()
        st.rerun()

    st.write("")

    # History section + hidden Manage popover
    hist_hdr_col, hist_mgr_col = st.columns([1.8, 1])
    with hist_hdr_col:
        st.markdown('<div class="sidebar-section-label">History</div>', unsafe_allow_html=True)
    with hist_mgr_col:
        if st.session_state["history"]:
            with st.popover("Manage", use_container_width=True):
                if st.session_state.get("active_review_id"):
                    if st.button("Delete Active Review", use_container_width=True):
                        st.session_state["history"] = [
                            h for h in st.session_state["history"]
                            if h["id"] != st.session_state["active_review_id"]
                        ]
                        st.session_state["active_review_id"] = (
                            st.session_state["history"][0]["id"]
                            if st.session_state["history"] else None
                        )
                        st.rerun()

                history_json = json.dumps([
                    {
                        "topic": h["topic"],
                        "timestamp": h["timestamp"],
                        "focus": h.get("focus", ""),
                        "files": [f["name"] for f in h.get("files", [])],
                        "notes": h.get("notes", ""),
                        "report": h["report"],
                        "verification": h["verification"]
                    }
                    for h in st.session_state["history"]
                ], indent=2)

                st.download_button(
                    "Export Archive (.json)",
                    data=history_json,
                    file_name="research_history.json",
                    mime="application/json",
                    use_container_width=True
                )

                if st.button("Clear All History", use_container_width=True):
                    st.session_state["history"] = []
                    st.session_state["active_review_id"] = None
                    st.rerun()

    # Optional filter when multiple reviews exist
    history_filter = ""
    if len(st.session_state["history"]) > 1:
        history_filter = st.text_input(
            "Filter history",
            placeholder="Filter past reviews...",
            label_visibility="collapsed"
        ).strip().lower()

    if not st.session_state["history"]:
        st.caption("Completed literature reviews appear here.")
    else:
        filtered_items = [
            item for item in st.session_state["history"]
            if not history_filter or history_filter in item["topic"].lower()
        ]
        if not filtered_items:
            st.caption("No matching reviews.")
        for item in filtered_items:
            is_active = (item["id"] == st.session_state.get("active_review_id"))
            label = f"> {item['topic']}" if is_active else item["topic"]
            if st.button(
                label,
                key=f"hist_{item['id']}",
                help=f"{item['timestamp']} • {len(item.get('files', []))} file(s)",
                use_container_width=True
            ):
                st.session_state["active_review_id"] = item["id"]
                st.rerun()

    st.divider()

    # Hidden / collapsed Researcher Preferences (Zero API Key or Temperature clutter)
    with st.expander("Research Preferences", expanded=False):
        source_scope = st.radio(
            "Evidence Sources",
            ["Hybrid (PDFs + Web)", "Uploaded PDFs Only", "Web Discovery Only"],
            index=0,
            help="Choose whether to combine uploaded manuscripts with live web literature or run a single source."
        )
        source_mode_key = (
            "hybrid" if "Hybrid" in source_scope
            else "docs_only" if "PDFs Only" in source_scope
            else "web_only"
        )

        review_focus = st.selectbox(
            "Review Focus",
            [
                "Comprehensive Literature Review",
                "Comparative Methodology Analysis",
                "Research Gaps & Open Problems",
                "Executive Summary Brief"
            ],
            index=0
        )

        writing_style = st.selectbox(
            "Writing Style",
            [
                "Academic Prose",
                "Structured Bullet Synthesis",
                "Critical Peer Review"
            ],
            index=0
        )

        search_depth = st.slider(
            "Web Discovery Depth",
            min_value=3,
            max_value=8,
            value=5,
            help="Number of web literature results gathered per topic."
        )

    # Collapsed Research Notes Scratchpad (linked to active review!)
    active_for_notes = get_active_review()
    with st.expander("Research Notes", expanded=False):
        if active_for_notes is not None:
            updated_notes = st.text_area(
                "Annotations for active review",
                value=active_for_notes.get("notes", ""),
                height=140,
                placeholder="Record hypotheses, key citations, or follow-up questions...",
                label_visibility="collapsed"
            )
            if updated_notes != active_for_notes.get("notes", ""):
                active_for_notes["notes"] = updated_notes

            if updated_notes.strip():
                st.download_button(
                    "Save Notes (.txt)",
                    data=f"Topic: {active_for_notes['topic']}\n\n{updated_notes}",
                    file_name="research_notes.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.caption("Select or complete a review to keep linked research notes.")


# --------------------------------------------------
# TOP HEADER & HIDDEN SAMPLE TOPICS
# --------------------------------------------------

hdr_col, sample_col = st.columns([4.4, 1.15], vertical_alignment="center")

with hdr_col:
    st.markdown("""
    <div class="hero-card">
        <div class="hero-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
            </svg>
        </div>
        <h1 class="hero-title">Autonomous Literature Review Assistance Agent</h1>
    </div>
    """, unsafe_allow_html=True)

with sample_col:
    with st.popover("Sample Topics", use_container_width=True):
        st.caption("Select a topic preset:")
        samples = [
            "Generative AI applications in clinical diagnostics",
            "Retrieval-augmented generation for hallucination mitigation",
            "Transformer attention mechanisms in long-context modeling",
            "Autonomous multi-agent workflows in scientific discovery"
        ]
        for idx, sample in enumerate(samples):
            if st.button(sample, key=f"sample_{idx}", use_container_width=True):
                st.session_state["topic_input"] = sample
                st.rerun()


# --------------------------------------------------
# RESEARCH INPUT WORKSPACE
# --------------------------------------------------

with st.container(border=True):
    topic = st.text_input(
        "Research Topic",
        key="topic_input",
        placeholder="Example: Generative AI applications in clinical diagnostics"
    )

    uploaded_files = st.file_uploader(
        "Research Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        disabled=(source_mode_key == "web_only"),
        key=f"uploader_{st.session_state['uploader_key']}",
        help="Optional in Web Discovery Only mode; required for manuscript indexing."
    )

    staged_meta = []
    if source_mode_key == "web_only":
        st.caption("Web Discovery Only mode active — literature will be gathered directly from live web sources.")
    elif uploaded_files:
        staged_meta = [inspect_pdf_metadata(f) for f in uploaded_files]
        chips_html = "".join(
            f'<span class="staged-chip">{html.escape(m["name"])}'
            f'<span class="staged-meta">{m["pages"]} pg &middot; {m["size_str"]}</span></span>'
            for m in staged_meta
        )
        st.markdown(f'<div class="staged-files">{chips_html}</div>', unsafe_allow_html=True)

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

    elif source_mode_key != "web_only" and not uploaded_files:

        st.error("Please upload at least one PDF research paper (or select 'Web Discovery Only' in Sidebar -> Research Preferences).")

    else:

        try:

            with st.status("Running literature review...", expanded=True) as status:

                active_vs = None
                if source_mode_key != "web_only" and uploaded_files:
                    st.write("Parsing uploaded PDF manuscripts...")
                    temp_dir = tempfile.mkdtemp()
                    uploaded_paths = []
                    for uploaded_file in uploaded_files:
                        fname = os.path.basename(uploaded_file.name)
                        file_path = os.path.join(temp_dir, fname)
                        with open(file_path, "wb") as file:
                            file.write(uploaded_file.getbuffer())
                        uploaded_paths.append(file_path)

                    st.write("Indexing semantic chunks into vector store...")
                    active_vs = ingest_pdfs(pdf_paths=uploaded_paths)

                st.write("Executing document retrieval, web search, verification, and synthesis...")
                result = research_workflow.invoke({
                    "topic": topic.strip(),
                    "rag_result": "",
                    "web_result": "",
                    "verification": "",
                    "report": "",
                    "max_results": search_depth,
                    "source_mode": source_mode_key,
                    "review_focus": review_focus,
                    "writing_style": writing_style,
                    "vectorstore": active_vs
                })

                report = result["report"]

                st.write("Formatting Markdown and PDF deliverables...")
                markdown_path = os.path.join(OUTPUT_DIR, "research_report.md")
                with open(markdown_path, "w", encoding="utf-8") as file:
                    file.write(report)

                pdf_bytes = create_pdf(report, topic.strip())
                pdf_path = os.path.join(OUTPUT_DIR, "research_report.pdf")
                with open(pdf_path, "wb") as file:
                    file.write(pdf_bytes)

                review_entry = {
                    "id": uuid.uuid4().hex[:8],
                    "topic": topic.strip(),
                    "timestamp": datetime.now().strftime("%H:%M • %d %b %Y"),
                    "mode": source_scope,
                    "focus": review_focus,
                    "style": writing_style,
                    "files": staged_meta,
                    "report": report,
                    "pdf": pdf_bytes,
                    "rag_result": result.get("rag_result", ""),
                    "web_result": result.get("web_result", ""),
                    "verification": result.get("verification", ""),
                    "vectorstore": active_vs,
                    "qa_log": [],
                    "notes": ""
                }

                st.session_state["history"].insert(0, review_entry)
                st.session_state["active_review_id"] = review_entry["id"]

                status.update(label="Research completed.", state="complete", expanded=False)
                st.rerun()

        except Exception as error:

            err_msg = str(error)
            if "no active llm backend" in err_msg.lower() or "ollama" in err_msg.lower() or "connection" in err_msg.lower():
                st.error(
                    "Unable to reach the inference service. Locally, ensure Ollama is running (`ollama serve`). "
                    "On Streamlit Cloud, ensure `GROQ_API_KEY` is configured in App Secrets."
                )
            else:
                st.error("The research workflow failed during execution.")

            with st.expander("Technical error details"):
                st.code(err_msg)


# --------------------------------------------------
# GENERATED REPORT CONSOLE (ACTIVE OR HISTORY)
# --------------------------------------------------

active_review = get_active_review()

if active_review:

    word_count = len(active_review["report"].split())
    read_mins = max(1, math.ceil(word_count / 220))
    doc_count = len(active_review.get("files", []))

    st.write("")
    with st.container(border=True):

        header_left, header_right = st.columns([2.2, 1.4])
        with header_left:
            st.subheader(active_review["topic"])
            st.caption(
                f"{active_review.get('focus', 'Literature Review')} • "
                f"{active_review.get('style', 'Academic Prose')} • "
                f"{active_review['timestamp']}"
            )

        with header_right:
            act_col1, act_col2, act_col3 = st.columns([1, 1, 0.95])
            with act_col1:
                md_export = active_review["report"]
                if active_review.get("notes", "").strip():
                    md_export += f"\n\n---\n## Researcher Notes\n{active_review['notes'].strip()}"
                st.download_button(
                    label="Markdown",
                    data=md_export,
                    file_name="research_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with act_col2:
                st.download_button(
                    label="PDF",
                    data=active_review["pdf"],
                    file_name="research_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with act_col3:
                with st.popover("More", use_container_width=True):
                    if active_review.get("files"):
                        st.caption("Indexed Manuscripts:")
                        for f in active_review["files"]:
                            st.markdown(f"- `{f['name']}` ({f['pages']} pg)")
                        st.divider()
                    st.download_button(
                        "Raw Text (.txt)",
                        data=active_review["report"],
                        file_name="research_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    if st.button("Hide View", use_container_width=True):
                        st.session_state["active_review_id"] = None
                        st.rerun()

        # Micro metrics strip
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Word Count</div>
                <div class="metric-value">{word_count:,}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Est. Read Time</div>
                <div class="metric-value">{read_mins} min</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Manuscripts</div>
                <div class="metric-value">{doc_count} PDF(s)</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Source Scope</div>
                <div class="metric-value">{html.escape(active_review.get('mode', 'Hybrid').split(' ')[0])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_report, tab_verification, tab_evidence, tab_qa = st.tabs([
            "Report",
            "Cross-Verification",
            "Sources & Retrieval",
            "Ask Manuscripts"
        ])

        with tab_report:
            with st.popover("Search in Report"):
                keyword = st.text_input(
                    "Highlight matching lines containing:",
                    placeholder="e.g., accuracy, limitation, dataset"
                ).strip()
                if keyword:
                    matches = [
                        line.strip() for line in active_review["report"].splitlines()
                        if keyword.lower() in line.lower()
                    ]
                    if matches:
                        st.caption(f"Found {len(matches)} matching line(s):")
                        for m in matches[:12]:
                            st.markdown(f"> {m}")
                    else:
                        st.caption("No matching lines found.")

            st.markdown('<div class="report-body">', unsafe_allow_html=True)
            st.markdown(active_review["report"])
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_verification:
            if active_review.get("verification"):
                st.markdown(active_review["verification"])
            else:
                st.caption("No verification output recorded for this review.")

        with tab_evidence:
            if active_review.get("rag_result"):
                st.markdown("#### Uploaded Document Findings")
                st.markdown(active_review["rag_result"])
                st.divider()

            if active_review.get("web_result"):
                st.markdown("#### Web Search Findings")
                st.markdown(active_review["web_result"])

        with tab_qa:
            if active_review.get("vectorstore") is None:
                st.info("Upload and index PDFs in Hybrid or Uploaded PDFs Only mode to ask follow-up questions against your manuscripts.")
            else:
                st.caption("Query your indexed PDF papers directly for specific methods, sample sizes, metrics, or definitions.")
                qa_col1, qa_col2 = st.columns([4, 1])
                with qa_col1:
                    follow_up = st.text_input(
                        "Follow-up question",
                        placeholder="e.g., What datasets or evaluation metrics were reported in the papers?",
                        label_visibility="collapsed",
                        key=f"qa_input_{active_review['id']}"
                    )
                with qa_col2:
                    ask_btn = st.button("Ask", type="primary", use_container_width=True, key=f"qa_btn_{active_review['id']}")

                if ask_btn and follow_up.strip():
                    with st.spinner("Searching indexed manuscripts..."):
                        try:
                            ans = rag_agent(
                                question=follow_up.strip(),
                                vectorstore=active_review["vectorstore"]
                            )
                            active_review["qa_log"].insert(0, {
                                "q": follow_up.strip(),
                                "a": ans
                            })
                        except Exception as qa_err:
                            st.error(f"Could not answer question: {qa_err}")

                for entry in active_review.get("qa_log", []):
                    st.markdown(f"**Q: {html.escape(entry['q'])}**")
                    st.markdown(f'<div class="qa-answer">{entry["a"]}</div>', unsafe_allow_html=True)
                    st.write("")
