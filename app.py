import streamlit as st
from pdf_processor import extract_chunks_with_pages
from retrieval import build_index, hybrid_search, rerank
from llm import get_client, generate_answer, generate_quiz

st.title("📚 Study Notes Assistant")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client_llm = get_client(GROQ_API_KEY)

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    if "processed_filename" not in st.session_state or st.session_state["processed_filename"] != uploaded_file.name:

        chunks_with_pages = extract_chunks_with_pages(uploaded_file)
        with st.spinner("Processing your PDF... this may take a moment"):
            chunks_with_pages = extract_chunks_with_pages(uploaded_file)

        if len(chunks_with_pages) == 0:
            st.error("⚠️ No extractable text found in this PDF. It might be a scanned or image-based document. Please try a text-based PDF instead.")
            st.stop()

        collection, documents, metadatas, bm25 = build_index(chunks_with_pages)

        st.session_state["collection"] = collection
        st.session_state["documents"] = documents
        st.session_state["metadatas"] = metadatas
        st.session_state["bm25"] = bm25
        st.session_state["processed_filename"] = uploaded_file.name
        st.session_state["total_chunks"] = len(chunks_with_pages)

    st.success(f"PDF processed! ({st.session_state['total_chunks']} chunks created) Ask a question below.")

    query = st.text_input("Ask a question about your PDF:")
    alpha = st.slider("Semantic vs Keyword balance (higher = more semantic)", 0.0, 1.0, 0.6, 0.1)

    if query:
        collection = st.session_state["collection"]
        documents = st.session_state["documents"]
        metadatas = st.session_state["metadatas"]
        bm25 = st.session_state["bm25"]
       
        with st.spinner("Searching your document and generating an answer..."):
            shortlist = hybrid_search(query, collection, documents, metadatas, bm25, alpha=alpha, shortlist_size=10)
            top_chunks = rerank(query, shortlist, top_n=3)

        answer = generate_answer(client_llm, query, top_chunks)

        st.write("### Answer")
        st.write(answer)

        with st.expander("See retrieved chunks (hybrid + re-ranked, for debugging)"):
            for doc, score, meta in top_chunks:
                st.write(f"**Page {meta['page']}** (rerank score: {score:.2f}): {doc}")
                st.write("---")

    st.write("---")
    st.write("### 📝 Generate a Quiz")

    if st.button("Generate Quiz from this PDF"):
        with st.spinner("Generating your quiz..."):
            collection = st.session_state["collection"]
            all_data = collection.get()
            sample_chunks = all_data["documents"][:10]

        quiz = generate_quiz(client_llm, sample_chunks)
        st.write(quiz)