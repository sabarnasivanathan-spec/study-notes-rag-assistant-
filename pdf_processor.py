from pypdf import PdfReader

def extract_chunks_with_pages(uploaded_file, chunk_size=500, overlap=100):
    """Extract text from a PDF, chunked with overlap, with metadata attached."""
    reader = PdfReader(uploaded_file)
    chunks_with_pages = []
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        start = 0
        while start < len(page_text):
            end = start + chunk_size
            chunk = page_text[start:end]
            if chunk.strip():
                chunks_with_pages.append({
                    "text": chunk,
                    "page": page_num,
                    "source": uploaded_file.name,
                    "chunk_index": chunk_index
                })
                chunk_index += 1
            start += chunk_size - overlap

    return chunks_with_pages