import streamlit as st
import validators
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="LangChain RAG App",
    page_icon="🦜"
)

st.title("🦜 LangChain: RAG Q&A From YT or Website")


# ================= SIDEBAR =================

with st.sidebar:

    groq_api_key = st.text_input(
        "Enter Groq API Key",
        type="password"
    )


# ================= URL INPUT =================

generic_url = st.text_input(
    "Enter YouTube or Website URL"
)


# ================= EXTRACT YOUTUBE VIDEO ID =================

def extract_video_id(url):

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"

    match = re.search(pattern, url)

    return match.group(1) if match else None


# ================= LOAD CONTENT =================

def load_content(url):

    # ---------- YOUTUBE ----------

    if "youtube.com" in url or "youtu.be" in url:

        video_id = extract_video_id(url)

        transcript = YouTubeTranscriptApi().fetch(video_id)

        text = " ".join(
            [item.text for item in transcript]
        )

    # ---------- WEBSITE ----------

    else:

        loader = WebBaseLoader(url)

        docs = loader.load()

        text = ""

        for doc in docs:
            text += doc.page_content

    return text


# ================= CREATE VECTOR STORE =================

def create_vector_store(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        chunks,
        embeddings
    )

    return vectorstore


# ================= PROCESS URL =================

if st.button("Process URL"):

    # ---------- API KEY CHECK ----------

    if not groq_api_key:

        st.error("Please enter Groq API Key")

        st.stop()

    # ---------- URL VALIDATION ----------

    if not validators.url(generic_url):

        st.error("Please enter a valid URL")

        st.stop()

    try:

        # ---------- LOAD CONTENT ----------

        with st.spinner("Loading content..."):

            text = load_content(generic_url)

            # Optional limit
            text = text[:15000]

        # ---------- CREATE VECTOR STORE ----------

        with st.spinner("Creating Vector Store..."):

            vectorstore = create_vector_store(text)

            st.session_state.vectorstore = vectorstore

        # ---------- INITIALIZE LLM ----------

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_api_key
        )

        # ---------- SUMMARY PROMPT ----------

        summary_prompt = f"""
        Summarize the following content clearly
        in around 150 words.

        Content:
        {text[:4000]}
        """

        # ---------- GENERATE SUMMARY ----------

        with st.spinner("Generating Summary..."):

            summary_response = llm.invoke(
                summary_prompt
            )

            st.session_state.summary = (
                summary_response.content
            )

        # ---------- SUCCESS ----------

        st.success("RAG Pipeline Ready!")

    except Exception as e:

        st.error("An error occurred")

        st.exception(e)


# ================= SHOW SUMMARY =================

if "summary" in st.session_state:

    st.subheader("Summary")

    st.write(st.session_state.summary)


# ================= QUESTION INPUT =================

question = st.text_input(
    "Ask Questions From The Content"
)


# ================= QUESTION ANSWERING =================

if st.button("Ask Question"):

    # ---------- VECTOR STORE CHECK ----------

    if "vectorstore" not in st.session_state:

        st.error("Please process a URL first")

        st.stop()

    # ---------- EMPTY QUESTION CHECK ----------

    if not question:

        st.error("Please enter a question")

        st.stop()

    try:

        vectorstore = st.session_state.vectorstore

        # ---------- RETRIEVE RELEVANT CHUNKS ----------

        docs = vectorstore.similarity_search(
            question,
            k=3
        )

        # ---------- CREATE CONTEXT ----------

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # ---------- INITIALIZE LLM ----------

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_api_key
        )

        # ---------- QA PROMPT ----------

        prompt = f"""
        Answer the question using only the
        provided context.

        Context:
        {context}

        Question:
        {question}
        """

        # ---------- GENERATE ANSWER ----------

        with st.spinner("Generating Answer..."):

            response = llm.invoke(prompt)

        # ---------- STORE ANSWER ----------

        st.session_state.answer = response.content

    except Exception as e:

        st.error("Error while generating answer")

        st.exception(e)


# ================= SHOW ANSWER =================

if "answer" in st.session_state:

    st.subheader("Answer")

    st.write(st.session_state.answer)