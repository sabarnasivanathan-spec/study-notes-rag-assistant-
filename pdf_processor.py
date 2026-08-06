from pypdf import PdfReader

def extract_chunks_with_pages(uploaded_file, chunk_size=500):
    """Extract text from a PDF, chunked, with page numbers attached."""
    reader = PdfReader(uploaded_file)
    chunks_with_pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        page_chunks = [page_text[i:i+chunk_size] for i in range(0, len(page_text), chunk_size)]
        for chunk in page_chunks:
            if chunk.strip():
                chunks_with_pages.append({"text": chunk, "page": page_num})

    return chunks_with_pages