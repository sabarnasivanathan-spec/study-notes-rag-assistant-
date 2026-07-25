Today:built working RAG pipeline
-PDF extraction(pypdf)
-Chunking(fixed 500-char,needs improving)
-ChromaDB for vector storage/retrivel
-Groq API for answer generation
-Known issue: chunking cuts mid-sentence,hurts retrivel on tables
Next: citations,Streamlit UI,quiz generation