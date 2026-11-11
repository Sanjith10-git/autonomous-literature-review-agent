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
    Spacer
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
    page_title="Multi-Agent AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)


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
        textColor=colors.darkblue,
        spaceAfter=18
    )

    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    story = []

    story.append(
        Paragraph(
            "Multi-Agent AI Research Assistant",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Research Topic: " + html.escape(topic),
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 15))

    for line in report.splitlines():

        line = line.strip()

        if not line:
            story.append(Spacer(1, 6))
            continue

        if line.startswith("# "):

            story.append(
                Paragraph(
                    html.escape(line[2:]),
                    title_style
                )
            )

        elif line.startswith("## "):

            story.append(
                Paragraph(
                    html.escape(line[3:]),
                    heading_style
                )
            )

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

st.title("🤖 Multi-Agent AI Research Assistant")

st.write(
    "Research a topic using uploaded PDFs, "
    "web search, AI verification, and report generation."
)

st.info(
    "Upload your research papers, enter a topic, "
    "and click Start Research."
)


# --------------------------------------------------
# RESEARCH INPUTS
# --------------------------------------------------

topic = st.text_input(
    "Enter your research topic",
    placeholder="Example: Generative AI applications in healthcare"
)

uploaded_files = st.file_uploader(
    "Upload research papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# START RESEARCH
# --------------------------------------------------

if st.button("🚀 Start Research", type="primary"):

    if not topic.strip():

        st.error("Please enter a research topic.")

    elif not uploaded_files:

        st.error("Please upload at least one PDF research paper.")

    else:

        try:

            # Save uploaded PDFs.
            with st.spinner("Saving uploaded PDFs..."):

                for uploaded_file in uploaded_files:

                    file_path = os.path.join(
                        DATA_DIR,
                        os.path.basename(uploaded_file.name)
                    )

                    with open(file_path, "wb") as file:

                        file.write(uploaded_file.getbuffer())

            # Process PDFs and update ChromaDB.
            with st.spinner("Processing research documents..."):

                ingest_pdfs()

            # Run all agents using LangGraph.
            with st.spinner(
                "Running RAG, Web, Verification, and Report agents..."
            ):

                result = research_workflow.invoke({
                    "topic": topic.strip(),
                    "rag_result": "",
                    "web_result": "",
                    "verification": "",
                    "report": ""
                })

            report = result["report"]

            # Save Markdown report.
            markdown_path = os.path.join(
                OUTPUT_DIR,
                "research_report.md"
            )

            with open(
                markdown_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(report)

            # Generate and save PDF.
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

            st.success("Research completed successfully!")

        except Exception as error:

            st.error(
                "The research workflow failed. "
                "Check the terminal for details."
            )

            st.exception(error)


# --------------------------------------------------
# DISPLAY REPORT AND DOWNLOAD OPTIONS
# --------------------------------------------------

if "research_report" in st.session_state:

    st.divider()

    st.header("📄 Final Research Report")

    st.caption(
        "Topic: " + st.session_state["research_topic"]
    )

    st.markdown(
        st.session_state["research_report"]
    )

    st.download_button(
        label="⬇️ Download Research Report (.md)",
        data=st.session_state["research_report"],
        file_name="research_report.md",
        mime="text/markdown"
    )

    st.download_button(
        label="⬇️ Download Research Report (.pdf)",
        data=st.session_state["research_pdf"],
        file_name="research_report.pdf",
        mime="application/pdf"
    )

    st.success(
        "Your report is also saved in the outputs folder."
    )