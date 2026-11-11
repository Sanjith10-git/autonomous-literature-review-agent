import os
import io
import html
import json
import uuid
import tempfile
from datetime import datetime
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

from config.llm import get_groq_api_key, is_ollama_running, get_llm
from rag.ingest import ingest_pdfs
from workflow import research_workflow


# --------------------------------------------------
# PROJECT FOLDERS & SESSION DEFAULTS
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
# REFINED EDITORIAL DESIGN SYSTEM
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

    /* Main canvas proportions */
    .block-container {
        padding-top: 2.25rem;
        padding-bottom: 4rem;
        max-width: 920px;
    }

    /* Hide heading anchor icons */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }

    /* Sidebar visual hierarchy */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.22);
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding-bottom: 0.85rem;
        margin-bottom: 0.85rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.22);
    }
    .sidebar-brand-title {
        font-size: 0.96rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        line-height: 1.2;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        color: #64748B;
    }
    .sidebar-section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B;
        margin: 0.35rem 0 0.4rem 0;
    }

    /* Left-aligned clean sidebar history buttons */
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        justify-content: flex-start !important;
        text-align: left !important;
        border-radius: 7px !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 0.65rem !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button p {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        width: 100%;
    }

    /* Subtle hover-revealed popover triggers */
    [data-testid="stPopover"] > button {
        border-radius: 7px !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        opacity: 0.82;
        transition: opacity 0.15s ease, border-color 0.15s ease !important;
    }
    [data-testid="stPopover"] > button:hover {
        opacity: 1;
        border-color: #1D4ED8 !important;
        color: #1D4ED8 !important;
    }

    /* Editorial Top Header */
    .hero-wrap {
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
    }
    .hero-accent {
        width: 42px;
        height: 4px;
        background-color: #1E3A8A;
        border-radius: 2px;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin: 0 0 0.45rem 0;
    }
    .hero-subtitle {
        font-size: 0.96rem;
        opacity: 0.76;
        line-height: 1.55;
        margin: 0;
    }

    /* Elevated paper-like containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.26) !important;
        box-shadow: 0 8px 26px -8px rgba(15, 23, 42, 0.07) !important;
        padding: 0.35rem 0.25rem !important;
    }

    /* Clean text input & dropzone */
    [data-testid="stTextInput"] input {
        border-radius: 8px !important;
        padding: 0.58rem 0.8rem !important;
        font-size: 0.94rem !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #1D4ED8 !important;
        box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.14) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 10px !important;
        border: 1.5px dashed rgba(100, 116, 139, 0.4) !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #1D4ED8 !important;
    }

    /* Staged manuscript chips */
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
        padding: 0.3rem 0.65rem;
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
        padding: 0.6rem 1.2rem !important;
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
        border-color: rgba(148, 163, 184, 0.35) !important;
    }
    button[kind="secondary"]:hover {
        border-color: #1D4ED8 !important;
        color: #1D4ED8 !important;
    }

    /* Tabs accent */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1D4ED8 !important;
    }
    [data-baseweb="tab-highlight"] {
        background-color: #1D4ED8 !important;
    }

    /* Report metadata & reading flow */
    .meta-strip {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        color: #64748B;
        margin-top: -0.35rem;
        margin-bottom: 0.9rem;
    }
    .report-body {
        line-height: 1.72;
        font-size: 0.96rem;
    }
    .report-body blockquote {
        border-left: 3px solid #1D4ED8;
        padding-left: 1rem;
        opacity: 0.88;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PDF GENERATION FUNCTION
# --------------------------------------------------

def create_pdf(report: str, topic: str) -> bytes:
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
# SIDEBAR: HISTORY & HIDDEN CONTROLS
# --------------------------------------------------

configured_groq_key = get_groq_api_key()
local_ollama_up = is_ollama_running()

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1E3A8A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>
            <path d="M8 7h6"/>
            <path d="M8 11h8"/>
        </svg>
        <div>
            <div class="sidebar-brand-title">Literature Agent</div>
            <div class="sidebar-brand-sub">Research Workspace</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("+ New Review", use_container_width=True):
        start_new_review()
        st.rerun()

    st.write("")

    # History header + hidden Manage popover
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
                        "files": h["files"],
                        "report": h["report"],
                        "verification": h["verification"]
                    }
                    for h in st.session_state["history"]
                ], indent=2)

                st.download_button(
                    "Export All (.json)",
                    data=history_json,
                    file_name="research_history.json",
                    mime="application/json",
                    use_container_width=True
                )

                if st.button("Clear All History", use_container_width=True):
                    st.session_state["history"] = []
                    st.session_state["active_review_id"] = None
                    st.rerun()

    # History list
    if not st.session_state["history"]:
        st.caption("Completed reviews will be listed here.")
    else:
        for item in st.session_state["history"]:
            is_active = (item["id"] == st.session_state.get("active_review_id"))
            label = f"> {item['topic']}" if is_active else item["topic"]
            if st.button(
                label,
                key=f"hist_{item['id']}",
                help=f"{item['timestamp']} • {len(item['files'])} file(s)",
                use_container_width=True
            ):
                st.session_state["active_review_id"] = item["id"]
                st.rerun()

    st.divider()

    # Hidden / collapsed Settings panel
    with st.expander("Settings & Model Options", expanded=False):
        provider_choice = st.selectbox(
            "Inference Backend",
            ["Auto", "Groq Cloud", "Local Ollama"],
            index=0
        )
        provider_key = (
            "auto" if provider_choice == "Auto"
            else "groq" if provider_choice == "Groq Cloud"
            else "ollama"
        )

        if provider_key == "ollama" or (provider_key == "auto" and local_ollama_up):
            selected_model = st.selectbox(
                "Model",
                ["llama3.2", "mistral", "deepseek-r1"],
                index=0
            )
        else:
            selected_model = st.selectbox(
                "Model",
                ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
                index=0
            )

        groq_key_sidebar = st.text_input(
            "Groq API Key",
            value=configured_groq_key,
            type="password",
            placeholder="gsk_...",
            help="Optional locally when Ollama is running; required on cloud unless stored in Secrets."
        )

        include_web = st.checkbox("Include Web Search", value=True)
        search_depth = st.slider("Web Sources", min_value=3, max_value=8, value=5)
        temperature = st.slider("Temperature", min_value=0.0, max_value=0.5, value=0.0, step=0.1)

        if st.button("Test Engine Connection", use_container_width=True):
            try:
                test_llm = get_llm(
                    api_key=groq_key_sidebar.strip() or None,
                    model=selected_model,
                    provider=provider_key,
                    temperature=temperature
                )
                reply = test_llm.invoke("Respond with 'Ready' in one word.")
                msg = reply.content if hasattr(reply, "content") else str(reply)
                st.success(f"Connected: {msg[:30]}")
            except Exception as exc:
                st.error(f"Connection check failed: {str(exc)[:110]}")

    with st.expander("Workspace Info", expanded=False):
        env_label = "Local Ollama Active" if local_ollama_up else "Cloud Mode"
        st.caption(f"Environment: {env_label}")
        st.caption(f"Saved Reviews: {len(st.session_state['history'])}")
        st.caption("Local Export Path: `outputs/`")


# --------------------------------------------------
# MAIN WORKSPACE HEADER + HIDDEN SAMPLE TOPICS
# --------------------------------------------------

hdr_col, tool_col = st.columns([4.2, 1.1])

with hdr_col:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-accent"></div>
        <h1 class="hero-title">Autonomous Literature Review Assistance Agent</h1>
        <p class="hero-subtitle">
            Upload peer-reviewed manuscripts and specify a research topic to generate a verified, citation-backed synthesis report.
        </p>
    </div>
    """, unsafe_allow_html=True)

with tool_col:
    st.write("")
    st.write("")
    with st.popover("Sample Topics", use_container_width=True):
        st.caption("Load a topic into the workspace:")
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
# RESEARCH INPUT CARD
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
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    if uploaded_files:
        chips_html = "".join(
            f'<span class="staged-chip">{html.escape(f.name)}'
            f'<span class="staged-size">{len(f.getbuffer()) / 1024:.1f} KB</span></span>'
            for f in uploaded_files
        )
        st.markdown(f'<div class="staged-files">{chips_html}</div>', unsafe_allow_html=True)

    active_api_key = groq_key_sidebar.strip() or configured_groq_key
    needs_key_inline = (
        provider_key == "groq" or (provider_key == "auto" and not local_ollama_up)
    ) and not active_api_key

    if needs_key_inline:
        inline_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Enter your Groq key here or configure GROQ_API_KEY in App Secrets (also configurable in Sidebar Settings)."
        )
        if inline_key.strip():
            active_api_key = inline_key.strip()

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

    elif (provider_key == "groq" or (provider_key == "auto" and not local_ollama_up)) and not active_api_key:

        st.error(
            "Please provide a Groq API Key (above or inside Sidebar -> Settings & Model Options) "
            "or add `GROQ_API_KEY` to Streamlit Cloud Secrets."
        )

    else:

        try:

            with st.status("Running literature review...", expanded=True) as status:

                st.write("Saving uploaded PDF files...")
                temp_dir = tempfile.mkdtemp()
                uploaded_paths = []
                file_names = []
                for uploaded_file in uploaded_files:
                    fname = os.path.basename(uploaded_file.name)
                    file_path = os.path.join(temp_dir, fname)
                    with open(file_path, "wb") as file:
                        file.write(uploaded_file.getbuffer())
                    uploaded_paths.append(file_path)
                    file_names.append(fname)

                st.write("Indexing manuscripts into vector store...")
                ingest_pdfs(pdf_paths=uploaded_paths, provider=provider_key)

                st.write("Retrieving document evidence, searching web literature, and verifying claims...")
                result = research_workflow.invoke({
                    "topic": topic.strip(),
                    "rag_result": "",
                    "web_result": "",
                    "verification": "",
                    "report": "",
                    "api_key": active_api_key or None,
                    "provider": provider_key,
                    "model": selected_model,
                    "temperature": temperature,
                    "max_results": search_depth,
                    "include_web": include_web
                })

                report = result["report"]

                st.write("Generating Markdown and PDF copies...")
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
                    "files": file_names,
                    "report": report,
                    "pdf": pdf_bytes,
                    "rag_result": result.get("rag_result", ""),
                    "web_result": result.get("web_result", ""),
                    "verification": result.get("verification", "")
                }

                st.session_state["history"].insert(0, review_entry)
                st.session_state["active_review_id"] = review_entry["id"]

                status.update(label="Research completed.", state="complete", expanded=False)
                st.rerun()

        except Exception as error:

            err_msg = str(error)
            if "no active llm backend" in err_msg.lower() or "ollama" in err_msg.lower() or "connection" in err_msg.lower():
                st.error(
                    "Unable to reach an active LLM backend. Configure `GROQ_API_KEY` in App Secrets or Sidebar Settings, "
                    "or ensure Ollama is running locally."
                )
            else:
                st.error("The research workflow failed during execution.")

            with st.expander("Technical error details"):
                st.code(err_msg)


# --------------------------------------------------
# GENERATED REPORT (ACTIVE OR SELECTED FROM HISTORY)
# --------------------------------------------------

active_review = get_active_review()

if active_review:

    st.write("")
    with st.container(border=True):

        header_left, header_right = st.columns([2.2, 1.35])
        with header_left:
            st.subheader("Research Report")
            st.markdown(
                f'<div class="meta-strip">Topic: {html.escape(active_review["topic"])} &nbsp;|&nbsp; '
                f'{active_review["timestamp"]} &nbsp;|&nbsp; {len(active_review["files"])} manuscript(s)</div>',
                unsafe_allow_html=True
            )

        with header_right:
            act_col1, act_col2, act_col3 = st.columns([1, 1, 0.95])
            with act_col1:
                st.download_button(
                    label="Markdown",
                    data=active_review["report"],
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
                    st.caption("Manuscripts indexed:")
                    for fname in active_review["files"]:
                        st.markdown(f"- `{fname}`")
                    st.divider()
                    if st.button("Hide View", use_container_width=True):
                        st.session_state["active_review_id"] = None
                        st.rerun()

        tab_report, tab_verification, tab_evidence = st.tabs([
            "Report",
            "Cross-Verification",
            "Sources & Retrieval"
        ])

        with tab_report:
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
