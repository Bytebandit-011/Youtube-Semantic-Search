import streamlit as st
from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_community.document_loaders import YoutubeLoader


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="YouTube Video Q&A", page_icon="🎥", layout="wide")

st.title("🎥 YouTube Video Q&A with RAG")
st.markdown("Ask questions about any YouTube video transcript!")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("📹 Video Settings")
    video_id = st.text_input("Enter YouTube Video ID:", placeholder="e.g., dQw4w9WgXcQ")
    st.markdown("*Find the video ID in the URL: youtube.com/watch?v=**VIDEO_ID***")
    
    process_button = st.button("🔄 Process Video", type="primary")

if process_button:
    if not video_id:
        st.error("Please Enter valid YT ID")
    else:
        with st.spinner("Fetching transcript..."):
            try:
                # Build URL from video_id
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                # Load transcript using YoutubeLoader
                loader = YoutubeLoader.from_youtube_url(video_url)
                docs = loader.load()
                
                # Extract text from Document objects
                transcript = " ".join(doc.page_content for doc in docs)
                
                st.success(f"✅ Fetched transcript! ({len(transcript)} characters)")
                
                # Display transcript preview
                with st.expander("📝 View Transcript"):
                    st.text_area("Transcript", transcript, height=200, disabled=True)
 
                # Text Splitting
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.create_documents([transcript])
                st.info(f"📄 Created {len(chunks)} chunks")

                # Embeddings + Vector Store
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001"
                )   
                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                st.session_state.vector_store = vectorstore
                st.success("✅ Vector store created successfully!")

            except TranscriptsDisabled:
                st.error("❌ No captions available for this video")
            except Exception as e:
                st.error(f"❌ Error: {e}") 

st.markdown("---")
st.header("💬 Ask Questions")


if st.session_state.vector_store is not None:
    question = st.text_input("Enter your question:", placeholder="What is this video about?")
    ask_button = st.button("🔍 Get Answer", type="primary")

    if ask_button and question:
        with st.spinner("Searching for answer...."):
            try:
                # Retrieval
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity", 
                    search_kwargs={'k': 3}
                )
                
                # # Get relevant documents
                relevant_docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # LLM Setup
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                
                # Create prompt
                prompt_text = f"""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know or not available in the given video

                Context:
                {context}

                Question: {question}

                Answer:
                """

                
                # Generate answer
                answer = llm.invoke(prompt_text).content
                
                # Display answer
                st.markdown("### 🎯 Answer:")
                st.success(answer)
                
                # Show relevant context
                # with st.expander("📚 Relevant Context Used"):
                #     for i, doc in enumerate(relevant_docs, 1):
                #         st.markdown(f"**Chunk {i}:**")
                #         st.text(doc.page_content)
                #         st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
    
    elif ask_button:
        st.warning("⚠️ Please enter a question!")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and Google Gemini 🚀")
