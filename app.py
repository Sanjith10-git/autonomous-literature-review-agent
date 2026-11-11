import os
import io
import re
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
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

from config.llm import get_groq_api_key, is_ollama_running, get_llm, invoke_llm
from rag.ingest import ingest_pdfs, get_vectorstore
from agents.rag_agent import rag_agent
from workflow import research_workflow


# ── Directories ────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous Literature Review Assistance Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in {
    "history": [],
    "active_review_id": None,
    "topic_input": "",
    "uploader_key": 0,
    "notes": "",
    "qa_log": [],
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/*
 * PROFESSIONAL COLOUR PALETTE
 * Sidebar  : Deep charcoal  #111827 → #1F2937
 * Accent   : Emerald teal   #0F766E (primary)  /  #0D9488 (hover)
 * Canvas   : Warm white     #F9FAFB  cards: #FFFFFF
 * Text     : Near-black     #111827  muted: #6B7280
 * Border   : Cool grey      #E5E7EB
 * Danger   : Slate-rose     (alerts only)
 */

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #111827;
    background-color: #F9FAFB;
}
code, pre { font-family: 'JetBrains Mono', monospace !important; }

/* ── Canvas ── */
[data-testid="stAppViewContainer"] { background-color: #F9FAFB !important; }
[data-testid="stHeader"]           { background: transparent !important; }
.block-container {
    padding-top: 2.6rem !important;
    padding-bottom: 5rem !important;
    max-width: 960px !important;
}
[data-testid="stHeaderActionElements"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #111827 0%, #1F2937 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] *         { color: #9CA3AF !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3        { color: #F3F4F6 !important; }
[data-testid="stSidebar"] label     { color: #9CA3AF !important; }
[data-testid="stSidebar"] p         { color: #9CA3AF !important; }

.sidebar-brand {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.1rem 0 1.1rem 0;
    margin-bottom: 1.1rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.sidebar-logo {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem; flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(15,118,110,0.4);
}
.sidebar-brand-title { font-size: 0.94rem; font-weight: 700; color: #F9FAFB !important; line-height: 1.2; }
.sidebar-brand-sub   { font-size: 0.69rem; color: #6B7280 !important; letter-spacing: 0.04em; margin-top: 1px; }

.sidebar-label {
    font-size: 0.63rem; font-weight: 600; letter-spacing: 0.11em;
    text-transform: uppercase; color: #4B5563 !important;
    margin: 0.65rem 0 0.3rem 0;
}

[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    color: #9CA3AF !important;
    border-radius: 7px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.42rem 0.68rem !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    background: rgba(15,118,110,0.18) !important;
    border-color: rgba(15,118,110,0.45) !important;
    color: #D1FAE5 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button p {
    white-space: nowrap !important; overflow: hidden !important;
    text-overflow: ellipsis !important; width: 100%;
}

/* New Review CTA */
.new-btn button {
    background: linear-gradient(135deg, #0F766E, #0D9488) !important;
    border: none !important; color: #FFFFFF !important;
    font-weight: 600 !important; border-radius: 8px !important;
    box-shadow: 0 3px 10px rgba(15,118,110,0.35) !important;
}
.new-btn button:hover { filter: brightness(1.08) !important; }

/* ── Hero header ── */
.hero-wrap {
    background: linear-gradient(130deg, #111827 0%, #0F3460 50%, #134E4A 100%);
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 20px 50px -12px rgba(0,0,0,0.4);
}
.hero-wrap::before {
    content: '';
    position: absolute; top: -50px; right: -30px;
    width: 240px; height: 240px; border-radius: 50%;
    background: radial-gradient(circle, rgba(15,118,110,0.22) 0%, transparent 70%);
}
.hero-wrap::after {
    content: '';
    position: absolute; bottom: -70px; right: 100px;
    width: 180px; height: 180px; border-radius: 50%;
    background: radial-gradient(circle, rgba(13,148,136,0.14) 0%, transparent 70%);
}
.hero-title {
    font-size: 2.2rem; font-weight: 800;
    color: #FFFFFF; letter-spacing: -0.03em;
    line-height: 1.18; margin: 0 0 0.85rem 0;
    text-shadow: 0 2px 20px rgba(0,0,0,0.4);
}
.hero-subtitle {
    font-size: 0.97rem; color: rgba(255,255,255,0.78);
    line-height: 1.65; margin: 0; max-width: 520px;
}
/* ── White content cards ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 4px 16px -4px rgba(0,0,0,0.06) !important;
    background: #FFFFFF !important;
}

/* ── Input fields ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    border-radius: 8px !important;
    border-color: #D1D5DB !important;
    font-size: 0.93rem !important;
    padding: 0.58rem 0.85rem !important;
    background: #FAFAFA !important;
    color: #111827 !important;
    transition: all 0.18s ease !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #0F766E !important;
    box-shadow: 0 0 0 3px rgba(15,118,110,0.14) !important;
    background: #FFFFFF !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    border-radius: 10px !important;
    border: 2px dashed #D1D5DB !important;
    background: #FAFAFA !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #0F766E !important;
    background: #F0FAFA !important;
}

/* ── Manuscript chips ── */
.staged-files { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.4rem 0 0.5rem; }
.staged-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.22rem 0.65rem; border-radius: 20px;
    font-size: 0.78rem; font-weight: 500;
    background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46;
}
.staged-meta { font-size: 0.7rem; color: #059669; }

/* ── Primary button ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%) !important;
    border: none !important; color: #FFFFFF !important;
    border-radius: 9px !important; font-weight: 700 !important;
    font-size: 0.94rem !important; letter-spacing: 0.015em !important;
    padding: 0.62rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(15,118,110,0.38) !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover {
    filter: brightness(1.08) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 7px 20px rgba(15,118,110,0.42) !important;
}

/* ── Secondary buttons ── */
button[kind="secondary"] {
    border-radius: 8px !important; font-weight: 500 !important;
    border-color: #D1D5DB !important; color: #374151 !important;
    transition: all 0.15s ease !important;
}
button[kind="secondary"]:hover {
    border-color: #0F766E !important; color: #0F766E !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] { font-weight: 500 !important; font-size: 0.87rem !important; color: #6B7280 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #0F766E !important; font-weight: 600 !important; }
[data-baseweb="tab-highlight"] { background-color: #0F766E !important; height: 2px !important; }

/* ── Report reading view ── */
.report-body { line-height: 1.8; font-size: 0.95rem; color: #1F2937; }
.report-body h1, .report-body h2 { letter-spacing: -0.02em; color: #111827; }
.report-body h3 { color: #374151; }
.report-body blockquote {
    border-left: 3px solid #0F766E; padding-left: 1rem;
    color: #4B5563; font-style: italic; background: #F0FAFA;
    border-radius: 0 6px 6px 0; padding: 0.6rem 1rem;
}

/* ── Meta pills strip ── */
.meta-strip { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.1rem 0 1rem; }
.meta-pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.18rem 0.6rem; border-radius: 20px;
    font-size: 0.73rem; font-weight: 500;
    background: #F3F4F6; color: #4B5563; border: 1px solid #E5E7EB;
}

/* ── Q&A entries ── */
.qa-entry { margin-bottom: 1.1rem; }
.qa-q {
    font-weight: 600; font-size: 0.87rem;
    color: #0F766E; margin-bottom: 0.28rem;
}
.qa-a {
    font-size: 0.87rem; line-height: 1.68;
    color: #374151; padding-left: 0.8rem;
    border-left: 2px solid #A7F3D0;
}

/* ── Popover trigger ── */
[data-testid="stPopover"] > button {
    border-radius: 7px !important; font-size: 0.8rem !important;
    font-weight: 500 !important; border: 1px solid #D1D5DB !important;
    color: #374151 !important; transition: all 0.15s ease !important;
}
[data-testid="stPopover"] > button:hover {
    border-color: #0F766E !important; color: #0F766E !important;
}

/* ── Info highlight box ── */
.info-box {
    background: #F0FAFA; border: 1px solid #A7F3D0;
    border-radius: 10px; padding: 0.9rem 1.1rem;
    font-size: 0.85rem; color: #065F46; line-height: 1.6;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def create_pdf(report: str, topic: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"],
                                 alignment=TA_CENTER,
                                 textColor=colors.HexColor("#1E3A8A"),
                                 fontSize=17, leading=22, spaceAfter=8)
    sub_style   = ParagraphStyle("S", parent=styles["Heading2"],
                                 textColor=colors.HexColor("#334155"),
                                 fontSize=11, leading=14, spaceAfter=10)
    h2_style    = styles["Heading2"]
    h3_style    = styles["Heading3"]
    body_style  = styles["BodyText"]

    story = [
        Paragraph("Autonomous Literature Review Assistance Agent", title_style),
        Paragraph(f"Research Topic: {html.escape(topic)}", sub_style),
        HRFlowable(width="100%", thickness=0.6,
                   color=colors.HexColor("#CBD5E1"), spaceAfter=12),
    ]
    for line in report.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 5)); continue
        if line.startswith("# "):
            story.append(Paragraph(html.escape(line[2:]), h2_style))
        elif line.startswith("## "):
            story.append(Paragraph(html.escape(line[3:]), h2_style))
        elif line.startswith("### "):
            story.append(Paragraph(html.escape(line[4:]), h3_style))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph("&bull; " + html.escape(line[2:]), body_style))
            story.append(Spacer(1, 2))
        else:
            story.append(Paragraph(html.escape(line), body_style))
            story.append(Spacer(1, 4))
    doc.build(story)
    return buf.getvalue()


def word_count(text: str) -> int:
    return len(text.split())


def read_time(text: str) -> int:
    return max(1, math.ceil(word_count(text) / 230))


def get_active_review():
    aid = st.session_state.get("active_review_id")
    for item in st.session_state.get("history", []):
        if item["id"] == aid:
            return item
    return None


def start_new_review():
    st.session_state["active_review_id"] = None
    st.session_state["topic_input"] = ""
    st.session_state["uploader_key"] += 1
    st.session_state["qa_log"] = []
    st.session_state["notes"] = ""


def inspect_pdf(f) -> dict:
    try:
        reader = PdfReader(io.BytesIO(f.getbuffer()))
        pages = len(reader.pages)
        meta  = reader.metadata or {}
        title = meta.get("/Title", "").strip() or f.name
        author= meta.get("/Author", "").strip() or "—"
    except Exception:
        pages, title, author = 0, f.name, "—"
    size_kb = len(f.getbuffer()) / 1024
    size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
    return {"name": f.name, "title": title, "author": author,
            "pages": pages, "size_str": size_str}


# ── Sidebar ────────────────────────────────────────────────────────────────────
configured_key  = get_groq_api_key()
local_ollama_up = is_ollama_running()

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">📚</div>
        <div>
            <div class="sidebar-brand-title">Literature Agent</div>
            <div class="sidebar-brand-sub">Research Workspace</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="new-btn">', unsafe_allow_html=True)
    if st.button("＋ New Review", use_container_width=True):
        start_new_review(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # ── History ──
    h_col, m_col = st.columns([1.7, 1])
    with h_col:
        st.markdown('<div class="sidebar-label">Review History</div>', unsafe_allow_html=True)
    with m_col:
        if st.session_state["history"]:
            with st.popover("Manage", use_container_width=True):
                aid = st.session_state.get("active_review_id")
                if aid:
                    if st.button("🗑 Delete Active", use_container_width=True):
                        st.session_state["history"] = [
                            h for h in st.session_state["history"] if h["id"] != aid
                        ]
                        st.session_state["active_review_id"] = (
                            st.session_state["history"][0]["id"]
                            if st.session_state["history"] else None
                        )
                        st.rerun()

                st.download_button(
                    "⬇ Export All (.json)",
                    data=json.dumps([{
                        "topic": h["topic"], "timestamp": h["timestamp"],
                        "files": h["files"], "report": h["report"],
                        "verification": h.get("verification", "")
                    } for h in st.session_state["history"]], indent=2),
                    file_name="research_history.json",
                    mime="application/json",
                    use_container_width=True
                )
                if st.button("🧹 Clear All", use_container_width=True):
                    st.session_state["history"] = []
                    st.session_state["active_review_id"] = None
                    st.rerun()

    if not st.session_state["history"]:
        st.caption("Completed reviews appear here.")
    else:
        for item in st.session_state["history"]:
            is_active = item["id"] == st.session_state.get("active_review_id")
            label = ("▶ " if is_active else "") + item["topic"]
            if st.button(label, key=f"hist_{item['id']}",
                         help=f"{item['timestamp']} · {len(item['files'])} file(s)",
                         use_container_width=True):
                st.session_state["active_review_id"] = item["id"]
                st.session_state["qa_log"]  = item.get("qa_log", [])
                st.session_state["notes"]   = item.get("notes", "")
                st.rerun()

    st.divider()

    # ── Research Preferences ──
    with st.expander("⚙ Research Preferences", expanded=True):
        source_scope = st.selectbox(
            "Source Scope",
            ["Uploaded Manuscripts + Web", "Uploaded Manuscripts Only", "Web Discovery Only"],
            index=0
        )
        source_mode_key = {
            "Uploaded Manuscripts + Web": "both",
            "Uploaded Manuscripts Only":  "docs_only",
            "Web Discovery Only":          "web_only",
        }[source_scope]

        review_focus = st.selectbox(
            "Review Focus",
            ["Comprehensive Review", "Methodology Analysis",
             "Findings & Results", "Research Gaps", "Comparative Analysis"],
            index=0
        )

        writing_style = st.selectbox(
            "Writing Style",
            ["Academic Prose", "Structured Bullet Points",
             "Executive Summary", "Technical Report"],
            index=0
        )

        search_depth = st.slider("Web Sources", 3, 10, 5,
                                 help="Number of live web results to fetch.")

    with st.expander("ℹ System Status", expanded=False):
        env = "🟢 Local Ollama" if local_ollama_up else ("🔑 Groq Cloud" if configured_key else "⚠ No Backend")
        st.caption(f"Backend: {env}")
        st.caption(f"Sessions: {len(st.session_state['history'])} saved")
        st.caption("Exports saved to `outputs/`")


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <h1 class="hero-title">Autonomous Literature Review<br>Assistance Agent</h1>
    <p class="hero-subtitle">
        Upload peer-reviewed manuscripts and define a research topic to generate a
        verified, citation-backed synthesis report — powered by multi-agent AI.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Research Input Card ────────────────────────────────────────────────────────
with st.container(border=True):

    # Topic row + sample popover
    t_col, s_col = st.columns([3.6, 1])
    with t_col:
        st.markdown("**Research Topic**")
    with s_col:
        with st.popover("💡 Samples", use_container_width=True):
            st.caption("Click to load a topic:")
            for idx, sample in enumerate([
                "Generative AI in clinical diagnostics",
                "Retrieval-augmented generation for hallucination mitigation",
                "Transformer attention in long-context modeling",
                "Multi-agent workflows in scientific discovery",
                "Federated learning for privacy-preserving healthcare AI",
                "Large language models in drug discovery pipelines",
            ]):
                if st.button(sample, key=f"sample_{idx}", use_container_width=True):
                    st.session_state["topic_input"] = sample
                    st.rerun()

    topic = st.text_input(
        "Research Topic",
        key="topic_input",
        placeholder="e.g. Generative AI applications in clinical diagnostics",
        label_visibility="collapsed"
    )

    st.write("")

    # PDF uploader (hidden when web-only)
    if source_mode_key != "web_only":
        uploaded_files = st.file_uploader(
            "Research Papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}",
            help="Upload one or more peer-reviewed PDF manuscripts."
        )
    else:
        uploaded_files = []
        st.info("**Web Discovery Only** mode — no PDFs required. Live sources will be gathered automatically.")

    staged_meta = []
    if uploaded_files:
        staged_meta = [inspect_pdf(f) for f in uploaded_files]
        chips_html = "".join(
            f'<span class="staged-chip">📄 {html.escape(m["name"])}'
            f' <span class="staged-meta">{m["pages"]} pg · {m["size_str"]}</span></span>'
            for m in staged_meta
        )
        st.markdown(f'<div class="staged-files">{chips_html}</div>', unsafe_allow_html=True)

        # PDF metadata inspector
        with st.expander("📋 Manuscript Details", expanded=False):
            for m in staged_meta:
                st.markdown(
                    f"**{html.escape(m['title'])}**  \n"
                    f"Author: `{html.escape(m['author'])}` · Pages: `{m['pages']}` · Size: `{m['size_str']}`"
                )

    st.write("")
    run_clicked = st.button("🔍 Start Research", type="primary", use_container_width=True)


# ── Workflow Execution ─────────────────────────────────────────────────────────
if run_clicked:
    if not topic.strip():
        st.error("Please enter a research topic.")
    elif source_mode_key != "web_only" and not uploaded_files:
        st.error("Upload at least one PDF, or switch to **Web Discovery Only** in Research Preferences.")
    else:
        try:
            with st.status("Running literature review…", expanded=True) as status:

                active_vs = None
                if source_mode_key != "web_only" and uploaded_files:
                    st.write("📂 Parsing uploaded PDF manuscripts…")
                    temp_dir = tempfile.mkdtemp()
                    uploaded_paths = []
                    for uf in uploaded_files:
                        fp = os.path.join(temp_dir, os.path.basename(uf.name))
                        with open(fp, "wb") as fh:
                            fh.write(uf.getbuffer())
                        uploaded_paths.append(fp)

                    st.write("🧩 Indexing semantic chunks into vector store…")
                    active_vs = ingest_pdfs(pdf_paths=uploaded_paths)

                st.write("🤖 Running document retrieval, web search, verification & synthesis…")
                result = research_workflow.invoke({
                    "topic":        topic.strip(),
                    "rag_result":   "",
                    "web_result":   "",
                    "verification": "",
                    "report":       "",
                    "max_results":  search_depth,
                    "source_mode":  source_mode_key,
                    "review_focus": review_focus,
                    "writing_style": writing_style,
                    "vectorstore":  active_vs,
                })

                report = result["report"]

                st.write("📄 Generating Markdown and PDF deliverables…")
                md_path = os.path.join(OUTPUT_DIR, "research_report.md")
                with open(md_path, "w", encoding="utf-8") as fh:
                    fh.write(report)

                pdf_bytes = create_pdf(report, topic.strip())
                pdf_path = os.path.join(OUTPUT_DIR, "research_report.pdf")
                with open(pdf_path, "wb") as fh:
                    fh.write(pdf_bytes)

                entry = {
                    "id":           uuid.uuid4().hex[:8],
                    "topic":        topic.strip(),
                    "timestamp":    datetime.now().strftime("%H:%M · %d %b %Y"),
                    "mode":         source_scope,
                    "focus":        review_focus,
                    "style":        writing_style,
                    "files":        [m["name"] for m in staged_meta],
                    "staged_meta":  staged_meta,
                    "report":       report,
                    "pdf":          pdf_bytes,
                    "rag_result":   result.get("rag_result", ""),
                    "web_result":   result.get("web_result", ""),
                    "verification": result.get("verification", ""),
                    "vectorstore":  active_vs,
                    "qa_log":       [],
                    "notes":        "",
                }

                st.session_state["history"].insert(0, entry)
                st.session_state["active_review_id"] = entry["id"]
                st.session_state["qa_log"]  = []
                st.session_state["notes"]   = ""

                status.update(label="✅ Research completed.", state="complete", expanded=False)
                st.rerun()

        except Exception as error:
            err = str(error)
            el  = err.lower()
            if "groq_api_key" in el or "authentication" in el or "401" in el:
                st.error("**Groq API key error.** Verify your `GROQ_API_KEY` in Streamlit Cloud → App Settings → Secrets.")
            elif "no active" in el or "no active llm" in el:
                st.error("**No inference backend.** Add `GROQ_API_KEY` to Streamlit Secrets or run Ollama locally.")
            elif "rate_limit" in el or "429" in el:
                st.error("**Rate limit reached.** Wait a moment and try again.")
            elif "model" in el and ("not found" in el or "404" in el):
                st.error("**Model not found.** Check your Groq plan at console.groq.com.")
            else:
                st.error(f"**Research workflow error:** {err[:280]}")
            with st.expander("Technical details", expanded=False):
                st.code(err)


# ── Report Console ─────────────────────────────────────────────────────────────
active_review = get_active_review()

if active_review:

    st.write("")
    with st.container(border=True):

        # Header row
        h_left, h_right = st.columns([2.5, 1.5])
        with h_left:
            st.markdown(f"### 📑 Research Report")

            wc  = word_count(active_review["report"])
            rt  = read_time(active_review["report"])
            pills = [
                f"🗂 {active_review['topic'][:40]}{'…' if len(active_review['topic'])>40 else ''}",
                f"🕐 {active_review['timestamp']}",
                f"📄 {len(active_review['files'])} manuscript(s)",
                f"📝 {wc:,} words",
                f"⏱ ~{rt} min read",
                f"🎯 {active_review.get('focus','—')}",
            ]
            pills_html = "".join(f'<span class="meta-pill">{p}</span>' for p in pills)
            st.markdown(f'<div class="meta-strip">{pills_html}</div>', unsafe_allow_html=True)

        with h_right:
            dl1, dl2, dl3 = st.columns([1,1,1])
            with dl1:
                st.download_button("⬇ MD",  data=active_review["report"],
                                   file_name="report.md", mime="text/markdown",
                                   use_container_width=True)
            with dl2:
                st.download_button("⬇ PDF", data=active_review["pdf"],
                                   file_name="report.pdf", mime="application/pdf",
                                   use_container_width=True)
            with dl3:
                st.download_button("⬇ TXT", data=active_review["report"],
                                   file_name="report.txt", mime="text/plain",
                                   use_container_width=True)

        st.divider()

        # ── Tabs ──────────────────────────────────────────────────────────────
        (tab_report, tab_qa, tab_notes,
         tab_verify, tab_evidence) = st.tabs([
            "📄 Report",
            "💬 Ask Manuscripts",
            "📝 Research Notes",
            "✅ Verification",
            "🔍 Evidence & Sources",
        ])

        # ── Report tab ────────────────────────────────────────────────────────
        with tab_report:
            search_term = st.text_input(
                "Search in report", placeholder="Keyword to highlight…",
                label_visibility="visible", key="report_search"
            )

            report_text = active_review["report"]
            if search_term.strip():
                highlighted = re.sub(
                    f"({re.escape(search_term)})",
                    r'<mark style="background:#FDE68A;border-radius:3px;">\1</mark>',
                    report_text, flags=re.IGNORECASE
                )
                st.markdown(f'<div class="report-body">{highlighted}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="report-body">', unsafe_allow_html=True)
                st.markdown(report_text)
                st.markdown('</div>', unsafe_allow_html=True)

        # ── Q&A tab ───────────────────────────────────────────────────────────
        with tab_qa:
            st.markdown('<div class="info-box">Ask specific questions about the uploaded manuscripts. '
                        'The agent retrieves relevant passages and answers using your documents only.</div>',
                        unsafe_allow_html=True)

            qa_q = st.text_input("Your question", placeholder="What methodology did the authors use?",
                                 key="qa_input")
            if st.button("Ask", key="qa_ask", type="primary"):
                if not qa_q.strip():
                    st.warning("Enter a question first.")
                elif not active_review.get("vectorstore"):
                    st.warning("No vector store for this review — Q&A requires uploaded PDFs.")
                else:
                    with st.spinner("Searching manuscripts…"):
                        try:
                            answer = rag_agent(
                                qa_q.strip(),
                                vectorstore=active_review["vectorstore"]
                            )
                            log_entry = {"q": qa_q.strip(), "a": answer,
                                         "ts": datetime.now().strftime("%H:%M")}
                            st.session_state["qa_log"].append(log_entry)
                            # persist into history entry
                            for h in st.session_state["history"]:
                                if h["id"] == active_review["id"]:
                                    h.setdefault("qa_log", []).append(log_entry)
                        except Exception as e:
                            st.error(f"Q&A failed: {str(e)[:200]}")

            qa_log = st.session_state.get("qa_log", [])
            if qa_log:
                st.write("")
                for entry in reversed(qa_log):
                    st.markdown(
                        f'<div class="qa-entry">'
                        f'<div class="qa-q">🗨 {html.escape(entry["q"])}'
                        f'<span style="font-size:0.68rem;color:#94A3B8;font-weight:400;margin-left:0.5rem;">{entry["ts"]}</span></div>'
                        f'<div class="qa-a">{html.escape(entry["a"])}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                if st.button("Clear Q&A log", key="clear_qa"):
                    st.session_state["qa_log"] = []
                    for h in st.session_state["history"]:
                        if h["id"] == active_review["id"]:
                            h["qa_log"] = []
                    st.rerun()

        # ── Notes tab ─────────────────────────────────────────────────────────
        with tab_notes:
            st.markdown('<div class="info-box">Your personal research scratchpad — '
                        'observations, follow-up ideas, and annotations are saved per session.</div>',
                        unsafe_allow_html=True)

            notes_val = st.text_area(
                "Research Notes", value=st.session_state.get("notes", ""),
                height=320, placeholder="Add your annotations, follow-up questions, or key takeaways here…",
                label_visibility="collapsed", key="notes_editor"
            )
            n_col1, n_col2 = st.columns([1, 1])
            with n_col1:
                if st.button("💾 Save Notes", use_container_width=True):
                    st.session_state["notes"] = notes_val
                    for h in st.session_state["history"]:
                        if h["id"] == active_review["id"]:
                            h["notes"] = notes_val
                    st.success("Notes saved.")
            with n_col2:
                st.download_button(
                    "⬇ Export Notes (.txt)",
                    data=notes_val,
                    file_name=f"notes_{active_review['id']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        # ── Verification tab ──────────────────────────────────────────────────
        with tab_verify:
            if active_review.get("verification"):
                st.markdown(active_review["verification"])
            else:
                st.caption("No verification data for this review.")

        # ── Evidence tab ──────────────────────────────────────────────────────
        with tab_evidence:
            if active_review.get("staged_meta"):
                st.markdown("#### Indexed Manuscripts")
                for m in active_review["staged_meta"]:
                    with st.expander(f"📄 {m['title']}", expanded=False):
                        st.markdown(f"- **File:** `{m['name']}`")
                        st.markdown(f"- **Author:** {m['author']}")
                        st.markdown(f"- **Pages:** {m['pages']} · **Size:** {m['size_str']}")
                st.divider()

            if active_review.get("rag_result"):
                st.markdown("#### Document Retrieval Findings")
                st.markdown(active_review["rag_result"])
                st.divider()

            if active_review.get("web_result"):
                st.markdown("#### Live Web Search Findings")
                st.markdown(active_review["web_result"])
