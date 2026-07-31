import streamlit as st
from pypdf import PdfReader
import chromadb
from groq import Groq
from rank_bm25 import BM25Okapi

st.title("📚 Study Notes Assistant")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client_llm = Groq(api_key=GROQ_API_KEY)

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
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

        # Build BM25 index over the same chunks (for hybrid search)
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        st.session_state["collection"] = collection
        st.session_state["documents"] = documents
        st.session_state["metadatas"] = metadatas
        st.session_state["bm25"] = bm25
        st.session_state["processed_filename"] = uploaded_file.name
        st.session_state["total_chunks"] = len(chunks_with_pages)

    st.success(f"PDF processed! ({st.session_state['total_chunks']} chunks created) Ask a question below.")

    # --- Q&A Section ---
    query = st.text_input("Ask a question about your PDF:")

    # Let the user tune the balance (optional, but nice for demoing)
    alpha = st.slider("Semantic vs Keyword balance (higher = more semantic)", 0.0, 1.0, 0.6, 0.1)

    if query:
        collection = st.session_state["collection"]
        all_documents = st.session_state["documents"]
        all_metadatas = st.session_state["metadatas"]
        bm25 = st.session_state["bm25"]

        # --- Semantic search: get ALL chunks ranked ---
        semantic_results = collection.query(query_texts=[query], n_results=len(all_documents))
        sem_docs = semantic_results['documents'][0]
        sem_distances = semantic_results['distances'][0]
        sem_metas = semantic_results['metadatas'][0]

        max_distance = max(sem_distances) if max(sem_distances) > 0 else 1
        semantic_scores = {}
        doc_to_meta = {}
        for doc, dist, meta in zip(sem_docs, sem_distances, sem_metas):
            similarity = 1 - (dist / max_distance)
            semantic_scores[doc] = similarity
            doc_to_meta[doc] = meta

        # --- Keyword search: BM25 over all chunks ---
        tokenized_query = query.lower().split()
        bm25_raw_scores = bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1
        bm25_scores = {doc: score / max_bm25 for doc, score in zip(all_documents, bm25_raw_scores)}

        # --- Combine scores ---
        combined = []
        for doc in all_documents:
            sem_score = semantic_scores.get(doc, 0)
            kw_score = bm25_scores.get(doc, 0)
            final_score = alpha * sem_score + (1 - alpha) * kw_score
            meta = doc_to_meta.get(doc, {"page": "?"})
            combined.append((doc, final_score, meta))

        combined.sort(key=lambda x: x[1], reverse=True)

        # Take top 3 after hybrid ranking
        top_chunks = combined[:3]

        context = "\n\n".join(
            f"[Page {meta['page']}]: {doc}"
            for doc, score, meta in top_chunks
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

        with st.expander("See retrieved chunks (hybrid ranked, for debugging)"):
            for doc, score, meta in top_chunks:
                st.write(f"**Page {meta['page']}** (hybrid score: {score:.2f}): {doc}")
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

        st.write(quiz_response.choices[0].message.content)