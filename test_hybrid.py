from rank_bm25 import BM25Okapi
import chromadb

documents = [
    "The sky is blue during a clear day.",
    "Chapter 7 discusses machine learning algorithms.",
    "Bananas are a good source of potassium.",
    "Section 3.2 covers photosynthesis in plants."
]

query = "What does chapter 7 cover?"

# --- Semantic search (ChromaDB) ---
client = chromadb.Client()
collection = client.create_collection("hybrid_test_v2")
ids = [str(i) for i in range(len(documents))]
collection.add(documents=documents, ids=ids)

# Get semantic results for ALL documents (not just top-n), so we can score everything
semantic_results = collection.query(query_texts=[query], n_results=len(documents))

# ChromaDB returns "distances" (lower = more similar). Convert to a 0-1 "similarity" score.
semantic_docs = semantic_results['documents'][0]
semantic_distances = semantic_results['distances'][0]

# Build a lookup: document -> semantic similarity score (inverted distance, normalized)
max_distance = max(semantic_distances)
semantic_scores = {}
for doc, dist in zip(semantic_docs, semantic_distances):
    similarity = 1 - (dist / max_distance) if max_distance > 0 else 1
    semantic_scores[doc] = similarity

# --- Keyword search (BM25) ---
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
tokenized_query = query.lower().split()
bm25_raw_scores = bm25.get_scores(tokenized_query)

# Normalize BM25 scores to 0-1 range
max_bm25 = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1
bm25_scores = {doc: score / max_bm25 for doc, score in zip(documents, bm25_raw_scores)}

# --- Combine both scores ---
alpha = 0.5  # weight: 0.5 = equal balance between semantic and keyword

final_scores = []
for doc in documents:
    sem_score = semantic_scores.get(doc, 0)
    kw_score = bm25_scores.get(doc, 0)
    combined = alpha * sem_score + (1 - alpha) * kw_score
    final_scores.append((doc, combined, sem_score, kw_score))

# Sort by combined score, highest first
final_scores.sort(key=lambda x: x[1], reverse=True)

print("Hybrid search results (combined ranking):\n")
for doc, combined, sem, kw in final_scores:
    print(f"Combined: {combined:.2f}  |  Semantic: {sem:.2f}  |  Keyword: {kw:.2f}")
    print(f"  {doc}\n")