# 📚 Study Notes Assistant

An AI-powered study assistant that lets you upload PDFs and ask questions, get cited answers, and generate quizzes — built using Retrieval-Augmented Generation (RAG).

## Features
- 📄 Upload any text-based PDF (notes, articles, textbooks)
- 💬 Ask questions and get answers grounded in the document's actual content
- 📍 Answers include page-number citations for verification
- 📝 Auto-generate a 5-question multiple choice quiz from the uploaded material

## How It Works
1. **Text Extraction** — extracts text from the PDF, page by page
2. **Chunking** — splits text into smaller pieces while preserving page numbers
3. **Embedding & Storage** — converts chunks into vector embeddings and stores them in ChromaDB
4. **Retrieval** — finds the most relevant chunks for a user's question using semantic similarity search
5. **Generation** — passes retrieved chunks as context to an LLM (via Groq API) to generate a grounded, cited answer

## Tech Stack
- **Python**
- **Streamlit** — web interface
- **pypdf** — PDF text extraction
- **ChromaDB** — vector database for semantic search
- **Groq API** (Llama 3.1) — LLM for answer and quiz generation

## Running Locally

1. Clone this repo:
```bash
git clone https://github.com/sabarnasivanathan-spec/study-notes-rag-assistant-.git
cd study-notes-rag-assistant-
```

2. Install dependencies:
```bash
pip install -r requirements.tx
(Get a free key at [console.groq.com](https://console.groq.com))

3. Add your Groq API key — create a `.env` file in the project root:


4. Run the app:
```bash
streamlit run app.py
```

## Known Limitations
- Currently only supports text-based PDFs (scanned/image-based PDFs are not yet supported)
- Fixed-size chunking may occasionally split content awkwardly across chunks

## Future Improvements
- OCR support for scanned documents
- Smarter, section-aware chunking
- Multi-PDF support

---
Built as a learning project to explore RAG pipelines, vector databases, and LLM APIs.
```

