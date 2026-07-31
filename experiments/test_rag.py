from pypdf import PdfReader
import chromadb
from groq import Groq

# Step 1: Extract text page-by-page, tracking page numbers
reader = PdfReader("article.pdf")  # your actual filename
  
chunks_with_pages = []
chunk_size = 500

for page_num, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text()
    page_chunks = [page_text[i:i+chunk_size] for i in range(0, len(page_text), chunk_size)]
    for chunk in page_chunks:
        if chunk.strip():  # skip empty chunks
            chunks_with_pages.append({"text": chunk, "page": page_num})

print(f"Total chunks created: {len(chunks_with_pages)}")

# Step 2: Store in ChromaDB with page numbers as metadata
client_db = chromadb.Client()
collection = client_db.create_collection("pdf_notes_v2")

documents = [c["text"] for c in chunks_with_pages]
metadatas = [{"page": c["page"]} for c in chunks_with_pages]
ids = [str(i) for i in range(len(chunks_with_pages))]

collection.add(documents=documents, metadatas=metadatas, ids=ids)

# Step 3: Query
query = "What is the role of water in photosynthesis?"  # your actual question

results = collection.query(query_texts=[query], n_results=3)

print("\n--- Retrieved chunks (with pages) ---")
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"[Page {meta['page']}] {doc}")
    print("---")

# Step 4: Generate answer with Groq
client_llm = Groq(api_key="YOUR_API_KEY_HERE")

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

print("\n--- Final Answer ---")
print(response.choices[0].message.content)