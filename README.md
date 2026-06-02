# 🎥 YouTube RAG Q&A

An intelligent YouTube video Q&A application built with Streamlit, LangChain, and Google Gemini. Ask questions about any YouTube video and get AI-powered answers based on the video's transcript using Retrieval-Augmented Generation (RAG).

## ✨ Features

- 📹 **YouTube Transcript Extraction** - Automatically fetches transcripts from any YouTube video
- 🤖 **AI-Powered Q&A** - Uses Google's Gemini 2.5 Flash for intelligent responses
- 🔍 **Semantic Search** - FAISS vector store for efficient similarity search
- 💬 **Context-Aware Answers** - Responses based only on video content
- 🎨 **Clean UI** - Intuitive Streamlit interface

## 🖼️ Screenshots

### Main Interface
<!-- Add screenshot here -->
<img width="1918" height="966" alt="image" src="https://github.com/user-attachments/assets/1ef524ee-5cd3-48d3-9221-07514f59900d" />


### Video Processing
<!-- Add screenshot here -->
![Processing Video](screenshots/processing.png)

### Q&A in Action
<!-- Add screenshot here -->
![Q&A Example](screenshots/qa-example.png)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Google AI API Key ([Get it here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/youtube-rag-qa.git
   cd youtube-rag-qa
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📋 Usage

1. **Enter YouTube Video ID**
   - Find the video ID from the URL: `youtube.com/watch?v=VIDEO_ID`
   - Paste it in the sidebar input field

2. **Process the Video**
   - Click "🔄 Process Video" button
   - Wait for transcript extraction and vector store creation

3. **Ask Questions**
   - Type your question in the text input
   - Click "🔍 Get Answer"
   - Receive AI-generated answers based on the video content

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: Google Generative AI Embeddings (gemini-embedding-001)
- **Vector Store**: FAISS
- **Framework**: LangChain
- **Transcript API**: youtube-transcript-api

## 📦 Project Structure

```
youtube-rag-qa/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in repo)
├── .gitignore         # Git ignore file
└── README.md          # Project documentation
```

## 🔧 Configuration

### Chunk Settings
Modify text splitting parameters in `app.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust chunk size
    chunk_overlap=200     # Adjust overlap
)
```

### Retrieval Settings
Adjust number of retrieved chunks:
```python
retriever = st.session_state.vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={'k': 3}  # Number of chunks to retrieve
)
```

## ⚠️ Limitations

- Only works with videos that have available transcripts/captions
- Answers are limited to content explicitly mentioned in the transcript
- Processing time depends on video length

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the RAG framework
- [Google AI](https://ai.google.dev/) for Gemini models
- [Streamlit](https://streamlit.io/) for the web framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search



---

⭐ If you found this project helpful, please give it a star!
