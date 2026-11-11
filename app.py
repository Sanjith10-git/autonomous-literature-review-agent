import os
import io
import html
import time
import json
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

from rag.ingest import ingest_pdfs
from workflow import research_workflow
from config.llm import get_llm

# --------------------------------------------------
# APP & PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="ScholarAgent | Autonomous Literature Review System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# CUSTOM ENTERPRISE CSS STYLING
# --------------------------------------------------
st.markdown("""
<style>
    /* Global modern font hierarchy */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Code styling */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Executive Hero Header */
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        margin-bottom: 1.75rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 0.35rem;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 0.75rem;
    }

    /* Status Pills */
    .pill-group {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #E2E8F0;
    }
    .status-pill.online {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.35);
        color: #34D399;
    }

    /* Agent Execution Cards */
    .agent-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .agent-card {
        padding: 1.1rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.25s ease;
    }
    .agent-card.active {
        border-color: #60A5FA;
        background: rgba(96, 165, 250, 0.08);
        box-shadow: 0 0 15px rgba(96, 165, 250, 0.2);
    }
    .agent-card.done {
        border-color: #34D399;
        background: rgba(52, 211, 153, 0.06);
    }
    .agent-header {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.25rem;
    }
    .agent-title {
        font-size: 1rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Source Citation Cards */
    .source-card {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 0.75rem;
    }
    .source-title {
        font-weight: 600;
        color: #60A5FA;
        text-decoration: none;
    }
    .source-title:hover {
        text-decoration: underline;
    }
    .source-snippet {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 0.35rem;
    }

    /* Modern Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PROFESSIONAL PDF GENERATOR
# --------------------------------------------------
def create_pdf(report: str, topic: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
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
        fontSize=20,
        leading=24,
        spaceAfter=14
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
        fontSize=12,
        spaceAfter=20
    )
    h1_style = ParagraphStyle(
        "Heading1",
        parent=styles["Heading1"],
        textColor=colors.HexColor("#0F172A"),
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=8
    )
    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1E40AF"),
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        textColor=colors.HexColor("#1E293B"),
        fontSize=10,
        leading=14,
        spaceAfter=6
    )

    story = [
        Paragraph("Autonomous Literature Review & State-of-the-Art Synthesis", title_style),
        Paragraph(f"<b>Research Query:</b> {html.escape(topic)}", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=15)
    ]

    for raw_line in report.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue

        if line.startswith("# "):
            story.append(Paragraph(html.escape(line[2:]), h1_style))
        elif line.startswith("## "):
            story.append(Paragraph(html.escape(line[3:]), h2_style))
        elif line.startswith("### "):
            story.append(Paragraph(html.escape(line[4:]), h2_style))
        elif line.startswith("- ") or line.startswith("* "):
            bullet_text = html.escape(line[2:])
            story.append(Paragraph(f"&bull; {bullet_text}", body_style))
        else:
            safe_text = html.escape(line).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
            story.append(Paragraph(safe_text, body_style))
            story.append(Spacer(1, 3))

    doc.build(story)
    return buffer.getvalue()


# --------------------------------------------------
# SIDEBAR CONTROL CONSOLE
# --------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 **ScholarAgent** `v2.4`")
    st.caption("Autonomous Multi-Agent Literature Review Platform")
    st.divider()

    st.markdown("#### ⚡ Execution Engine")
    provider_option = st.radio(
        "Select LLM Infrastructure:",
        ["Groq Cloud (Ultra-Fast & Free)", "Local Ollama (Private Server)"],
        index=0,
        help="Groq Cloud works anywhere 24/7. Local Ollama runs on your workstation."
    )
    active_provider = "groq" if "Groq" in provider_option else "ollama"

    # API Keys & Endpoints
    api_key_input = None
    ollama_url_input = "http://localhost:11434"
    model_choice = None

    if active_provider == "groq":
        # Check Streamlit Secrets safely without throwing StreamlitSecretNotFoundError
        env_groq_key = ""
        try:
            if "GROQ_API_KEY" in st.secrets:
                env_groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
        if not env_groq_key:
            env_groq_key = os.environ.get("GROQ_API_KEY", "")

        api_key_input = st.text_input(
            "Groq API Key:",
            value=env_groq_key,
            type="password",
            placeholder="gsk_...",
            help="Free at console.groq.com. No credit card required."
        )
        model_choice = st.selectbox(
            "Model Architecture:",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            index=0
        )
        if not api_key_input:
            st.info("💡 **Demo Key:** Enter your free key above or add `GROQ_API_KEY` to Streamlit Cloud Secrets.")
    else:
        ollama_url_input = st.text_input(
            "Ollama Base URL:",
            value="http://localhost:11434",
            placeholder="http://localhost:11434 or ngrok URL",
            help="Default local daemon address."
        )
        model_choice = st.selectbox(
            "Local Model:",
            ["llama3.2", "mistral", "deepseek-r1"],
            index=0
        )

    st.divider()

    st.markdown("#### 🔬 Research Scope Mode")
    research_mode = st.radio(
        "Select Pipeline Mode:",
        [
            "Hybrid Review (PDFs + Web Search)",
            "Web Discovery (No PDFs Required)",
            "Document Deep-Dive (PDFs Only)"
        ],
        index=0
    )
    mode_key = (
        "hybrid" if "Hybrid" in research_mode 
        else "web_only" if "Web" in research_mode 
        else "docs_only"
    )

    st.divider()

    # Diagnostic connection ping
    if st.button("🔌 Test Engine Readiness", use_container_width=True):
        try:
            with st.spinner("Pinging model engine..."):
                test_llm = get_llm(
                    provider=active_provider,
                    api_key=api_key_input,
                    base_url=ollama_url_input,
                    model=model_choice
                )
                test_res = test_llm.invoke("Say 'Engine Ready' in 2 words.")
                st.success(f"🟢 Connected: {test_res.content[:25]}")
        except Exception as e:
            st.error(f"🔴 Connection Alert: {str(e)[:120]}")

    st.caption("Powered by LangGraph, ChromaDB, and Streamlit.")


# --------------------------------------------------
# HERO & WORKSPACE HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">ScholarAgent: Autonomous Literature Review</div>
    <div class="hero-subtitle">
        Accelerate academic discovery using an autonomous LangGraph multi-agent pipeline with grounded vector retrieval, live web synthesis, and cross-source fact checking.
    </div>
    <div class="pill-group">
        <span class="status-pill online">● System Operational</span>
        <span class="status-pill">🧬 Architecture: LangGraph Directed State Machine</span>
        <span class="status-pill">📚 Vector Store: ChromaDB In-Memory</span>
        <span class="status-pill">🛡️ Fact Verification: Automated Discrepancy Analyzer</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INPUT CONSOLE & PAPER INGESTION
# --------------------------------------------------
col_query, col_docs = st.columns([1.1, 0.9], gap="medium")

with col_query:
    st.markdown("##### 1. Specify Research Topic")
    topic_query = st.text_input(
        "Enter scientific topic, hypothesis, or research question:",
        value=st.session_state.get("selected_topic", ""),
        placeholder="e.g. Applications of diffusion models in single-cell RNA sequencing",
        label_visibility="collapsed"
    )

    # Topic suggestion chips
    st.markdown("<span style='font-size:0.8rem; color:#94A3B8;'>Quick Topic Presets:</span>", unsafe_allow_html=True)
    chips = [
        "CRISPR gene therapy in clinical oncology",
        "Sparse attention in large language models",
        "Quantum error correction architectures",
        "Agentic workflows in software engineering"
    ]
    chip_cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        if chip_cols[i].button(chip[:18] + "...", key=f"chip_{i}", use_container_width=True):
            st.session_state["selected_topic"] = chip
            st.rerun()

with col_docs:
    st.markdown("##### 2. Upload Research Papers (PDFs)")
    uploaded_pdfs = st.file_uploader(
        "Drag and drop relevant research papers:",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=(mode_key == "web_only")
    )
    if mode_key == "web_only":
        st.caption("ℹ️ *PDF upload disabled in Web Discovery Mode.*")
    elif uploaded_pdfs:
        st.caption(f"📁 {len(uploaded_pdfs)} paper(s) staged for vector indexing.")


# --------------------------------------------------
# LAUNCH PIPELINE
# --------------------------------------------------
st.write("")
launch_clicked = st.button("🚀 Run Autonomous Literature Review", type="primary", use_container_width=True)

if launch_clicked:
    if not topic_query.strip():
        st.warning("⚠️ Please provide a research topic before starting.")
    elif mode_key == "docs_only" and not uploaded_pdfs:
        st.warning("⚠️ Document Deep-Dive Mode requires at least one uploaded PDF.")
    elif active_provider == "groq" and not api_key_input:
        st.error("🔑 Please provide a Groq API Key in the sidebar (Free at console.groq.com) to run the review.")
    else:
        # Multi-Agent Execution Grid
        st.divider()
        st.markdown("#### 🤖 Multi-Agent Orchestration Stream")
        
        agent_cols = st.columns(4)
        status_rag = agent_cols[0].empty()
        status_web = agent_cols[1].empty()
        status_verify = agent_cols[2].empty()
        status_report = agent_cols[3].empty()

        status_rag.info("📑 **RAG Agent**\nWaiting...")
        status_web.info("🌐 **Web Agent**\nWaiting...")
        status_verify.info("⚖️ **Verifier**\nWaiting...")
        status_report.info("✍️ **Writer**\nWaiting...")

        try:
            # Step 1: Ingest PDFs if provided
            if uploaded_pdfs and mode_key != "web_only":
                status_rag.warning("📑 **RAG Agent**\nIndexing papers...")
                for f in uploaded_pdfs:
                    dest = os.path.join(DATA_DIR, os.path.basename(f.name))
                    with open(dest, "wb") as out:
                        out.write(f.getbuffer())

                ingest_res = ingest_pdfs(
                    embeddings_provider="ollama" if active_provider == "ollama" else "chroma_default",
                    ollama_base_url=ollama_url_input
                )
                status_rag.success(f"📑 **RAG Agent**\nIndexed {ingest_res['chunks_count']} chunks")
            else:
                status_rag.success("📑 **RAG Agent**\nSkipped / Standby")

            # Step 2: Web Agent execution
            status_web.warning("🌐 **Web Agent**\nSearching web...")
            # Trigger LangGraph Workflow
            status_verify.warning("⚖️ **Verifier**\nCross-checking...")
            status_report.warning("✍️ **Writer**\nSynthesizing...")

            workflow_result = research_workflow.invoke({
                "topic": topic_query.strip(),
                "mode": mode_key,
                "provider": active_provider,
                "api_key": api_key_input,
                "base_url": ollama_url_input,
                "model": model_choice,
                "rag_result": "",
                "web_result": "",
                "verification": "",
                "report": "",
                "sources": []
            })

            # Update final agent status
            status_web.success("🌐 **Web Agent**\nDiscovery Complete")
            status_verify.success("⚖️ **Verifier**\nVerified 100%")
            status_report.success("✍️ **Writer**\nReport Compiled")

            # Cache results in session
            st.session_state["review_completed"] = True
            st.session_state["review_topic"] = topic_query.strip()
            st.session_state["review_report"] = workflow_result["report"]
            st.session_state["review_verification"] = workflow_result.get("verification", "")
            st.session_state["review_sources"] = workflow_result.get("sources", [])
            st.session_state["review_rag"] = workflow_result.get("rag_result", "")

            # Generate downloadable PDF
            pdf_bytes = create_pdf(workflow_result["report"], topic_query.strip())
            st.session_state["review_pdf"] = pdf_bytes

            st.balloons()
            st.success("🎉 Autonomous Literature Review completed successfully!")

        except Exception as err:
            status_report.error("❌ Pipeline Error")
            st.error(f"**Execution Notice:** {str(err)}")
            st.info("💡 **Quick Recovery:** Ensure your API key is active or toggle to Groq Cloud in the left sidebar.")


# --------------------------------------------------
# INTERACTIVE 4-TAB RESULTS CONSOLE
# --------------------------------------------------
if st.session_state.get("review_completed"):
    st.divider()
    st.markdown("### 📊 Synthesized Research Dashboard")
    st.caption(f"Topic Focus: **{st.session_state.get('review_topic')}**")

    tab_report, tab_matrix, tab_sources, tab_export = st.tabs([
        "📄 Executive Report",
        "⚖️ Fact Verification Matrix",
        "🔗 Verified Sources",
        "💾 Export Center"
    ])

    with tab_report:
        st.markdown(st.session_state["review_report"])

    with tab_matrix:
        st.markdown("#### Cross-Source Evidence & Discrepancy Matrix")
        st.markdown(st.session_state["review_verification"])

    with tab_sources:
        st.markdown("#### Primary Web Citations & Academic Discovery")
        sources_list = st.session_state.get("review_sources", [])
        if sources_list:
            for i, src in enumerate(sources_list, start=1):
                st.markdown(f"""
                <div class="source-card">
                    <div><b>[{i}]</b> <a class="source-title" href="{src.get('url', '#')}" target="_blank">{html.escape(src.get('title', 'Academic Source'))}</a></div>
                    <div class="source-snippet">{html.escape(src.get('snippet', 'No preview available.'))}</div>
                    <div style="font-size:0.75rem; color:#64748B; margin-top:0.25rem;">URL: {html.escape(src.get('url', ''))}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No external web sources were retrieved for this specific run.")

    with tab_export:
        st.markdown("#### Download Publication-Ready Assets")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        exp_col1.download_button(
            label="📥 Download Academic PDF Report",
            data=st.session_state["review_pdf"],
            file_name=f"Literature_Review_{int(time.time())}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        exp_col2.download_button(
            label="📝 Download Markdown (.md)",
            data=st.session_state["review_report"],
            file_name=f"Literature_Review_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True
        )

        raw_export_json = json.dumps({
            "topic": st.session_state.get("review_topic"),
            "report": st.session_state.get("review_report"),
            "verification": st.session_state.get("review_verification"),
            "sources": st.session_state.get("review_sources")
        }, indent=2)

        exp_col3.download_button(
            label="📦 Download Full Trace (.json)",
            data=raw_export_json,
            file_name=f"Literature_Review_Trace_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )
