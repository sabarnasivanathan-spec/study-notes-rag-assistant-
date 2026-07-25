import streamlit as st
from pypdf import PdfReader
import chromadb
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

st.title("📚 Study Notes Assistant")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

client_llm = Groq(api_key=GROQ_API_KEY)

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Reprocess only if this is a new/different file
    if "processed_filename" not in st.session_state or st.session_state["processed_filename"] != uploaded_file.name:

        reader = PdfReader(uploaded_file)

        chunks_with_pages = []
        chunk_size = 500

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            page_chunks = [page_text[i:i+chunk_size] for i in range(0, len(page_text), chunk_size)]
            for chunk in page_chunks:
                if chunk.strip():
                    chunks_with_pages.append({"text": chunk, "page": page_num})
        if len(chunks_with_pages) == 0:
            st.error("⚠️ No extractable text found in this PDF. It might be a scanned or image-based document. Please try a text-based PDF instead.")
            st.stop()


        client_db = chromadb.Client()

        try:
            client_db.delete_collection("pdf_notes_app")
        except Exception:
            pass

        collection = client_db.create_collection("pdf_notes_app")

        documents = [c["text"] for c in chunks_with_pages]
        metadatas = [{"page": c["page"]} for c in chunks_with_pages]
        ids = [str(i) for i in range(len(chunks_with_pages))]

        collection.add(documents=documents, metadatas=metadatas, ids=ids)

        st.session_state["collection"] = collection
        st.session_state["processed_filename"] = uploaded_file.name
        st.session_state["total_chunks"] = len(chunks_with_pages)

    st.success(f"PDF processed! ({st.session_state['total_chunks']} chunks created) Ask a question below.")

    # --- Q&A Section ---
    query = st.text_input("Ask a question about your PDF:")

    if query:
        collection = st.session_state["collection"]
        results = collection.query(query_texts=[query], n_results=3)

        context = "\n\n".join(
            f"[Page {meta['page']}]: {doc}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        )

        prompt = f"""Answer the question based only on the context below. Mention which page(s) your answer comes from.

Context:
{context}

Question: {query}

Answer:"""

        response = client_llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers based only on the given context and cites page numbers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        st.write("### Answer")
        st.write(response.choices[0].message.content)

        with st.expander("See retrieved chunks (for debugging)"):
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                st.write(f"**Page {meta['page']}:** {doc}")
                st.write("---")

    # --- Quiz Section ---
    st.write("---")
    st.write("### 📝 Generate a Quiz")

    if st.button("Generate Quiz from this PDF"):
        collection = st.session_state["collection"]

        all_data = collection.get()
        sample_chunks = all_data["documents"][:10]
        quiz_context = "\n\n".join(sample_chunks)

        quiz_prompt = f"""Based on the following content, create a 5-question multiple choice quiz.
For each question, provide 4 options (A, B, C, D) and clearly indicate the correct answer at the end.

Content:
{quiz_context}

Quiz:"""

        quiz_response = client_llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates quizzes from study material."},
                {"role": "user", "content": quiz_prompt}
            ],
            temperature=0.5
        )
