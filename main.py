import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_community.document_loaders import YoutubeLoader
import os

# Page configuration
st.set_page_config(page_title="YouTube Video Q&A", page_icon="🎥", layout="wide")

st.title("🎥 YouTube Video Q&A with RAG")
st.markdown("Ask questions about any YouTube video transcript!")

# Sidebar for settings
with st.sidebar:
    st.header("🔑 API Settings")
    user_api_key = st.text_input("Enter your Google API Key:", type="password")
    st.markdown("*Your key is used only during this session.*")

    st.header("📹 Video Settings")
    video_id = st.text_input("Enter YouTube Video ID:", placeholder="e.g., dQw4w9WgXcQ")
    process_button = st.button("🔄 Process Video", type="primary")

# Store vector store in session
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if process_button:
    if not video_id or not user_api_key:
        st.error("Please enter both a valid YouTube ID and your API key.")
    else:
        with st.spinner("Fetching transcript..."):
            try:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                loader = YoutubeLoader.from_youtube_url(video_url)
                docs = loader.load()
                transcript = " ".join(doc.page_content for doc in docs)

                st.success(f"✅ Fetched transcript! ({len(transcript)} characters)")

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=user_api_key   # use user-provided key
                )
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.vector_store = vectorstore
                st.success("✅ Vector store created successfully!")

            except TranscriptsDisabled:
                st.error("❌ No captions available for this video")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Question answering
st.header("💬 Ask Questions")
if st.session_state.vector_store is not None and user_api_key:
    question = st.text_input("Enter your question:", placeholder="What is this video about?")
    ask_button = st.button("🔍 Get Answer", type="primary")

    if ask_button and question:
        with st.spinner("Searching for answer..."):
            try:
                retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 3})
                relevant_docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=user_api_key)

                prompt_text = f"""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                Context:
                {context}

                Question: {question}

                Answer:
                """

                answer = llm.invoke(prompt_text).content
                st.markdown("### 🎯 Answer:")
                st.success(answer)

            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
