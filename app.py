import streamlit as st
import validators
import re

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------- RAG IMPORTS ----------------
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configure Streamlit page
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

# Main title
st.title("LangChain: Summarize Text From YT or Website")

st.subheader("Summarize URL with RAG")

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

# URL input
generic_url = st.text_input("Enter YouTube or Website URL")


# Extract YouTube Video ID
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None


# ---------------- RAG FUNCTIONS ----------------

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
def create_vector_store(text_chunks):
    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embedding_model
    )
    return vector_store


# Retrieve relevant chunks
def retrieve_relevant_chunks(vector_store, query):
    docs = vector_store.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])


# ------------------------------------------------


# Session state initialization
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False

if "final_summary" not in st.session_state:
    st.session_state.final_summary = ""

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# Button
if st.button("Summarize the Content from YT or Website"):

    # Check API key
    if not groq_api_key:
        st.error("Please enter your Groq API key.")
        st.stop()

    # Validate URL
    if not validators.url(generic_url):
        st.error("Please enter a valid URL.")
        st.stop()

    try:

        # Loading content
        with st.spinner("Loading content..."):

            # YouTube URL
            if "youtube.com" in generic_url or "youtu.be" in generic_url:

                # Extract video ID
                video_id = extract_video_id(generic_url)

                # Fetch transcript
                transcript = YouTubeTranscriptApi().fetch(video_id)

                # Convert transcript to text
                text = " ".join([item.text for item in transcript])

            # Website URL
            else:

                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0"}
                )

                docs = loader.load()

                text = ""

                for doc in docs:
                    text += doc.page_content

        # Limit text size
        text = text[:5000]

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        texts = text_splitter.split_text(text)

        # Reduce chunks for faster response
        texts = texts[:4]

        # Create vector store
        st.session_state.vector_store = create_vector_store(texts)

        # Initialize LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_api_key
        )

        summaries = []

        # Generate summaries
        with st.spinner("Generating summary using RAG..."):

            for chunk in texts:

                # Retrieve relevant context
                context = retrieve_relevant_chunks(
                    st.session_state.vector_store,
                    chunk
                )

                # RAG Prompt
                prompt = f"""
                Use the following retrieved context to generate
                a better summary.

                Context:
                {context}

                Current Chunk:
                {chunk}

                Give a concise summary in about 80 words.
                """

                # LLM response
                response = llm.invoke(prompt)

                summaries.append(response.content)

        # Final summary
        final_summary = " ".join(summaries)

        # Store in session state
        st.session_state.final_summary = final_summary
        st.session_state.summary_generated = True

    # Error handling
    except Exception as e:
        st.error("An error occurred while processing.")
        st.exception(e)


# ---------------- DISPLAY SUMMARY ----------------

if st.session_state.summary_generated:

    st.success("Summary Generated Successfully!")

    st.write(st.session_state.final_summary)

    # ---------------- QUESTION ANSWERING ----------------

    st.subheader("Ask Questions from Content")

    user_question = st.text_input(
        "Ask a question based on the content"
    )

    if user_question:

        try:

            # Initialize LLM
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=groq_api_key
            )

            with st.spinner("Generating Answer..."):

                # Retrieve relevant chunks
                context = retrieve_relevant_chunks(
                    st.session_state.vector_store,
                    user_question
                )

                # QA Prompt
                qa_prompt = f"""
                Answer the question ONLY using the context below.

                Context:
                {context}

                Question:
                {user_question}
                """

                # Generate answer
                answer = llm.invoke(qa_prompt)

                # Display answer
                st.write(answer.content)

        except Exception as e:
            st.error("Error while generating answer.")
            st.exception(e)
