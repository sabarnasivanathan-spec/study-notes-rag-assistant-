import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Load the re-ranking model once (shared across calls)
_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def build_index(chunks_with_pages):
    """Build ChromaDB collection + BM25 index from chunks."""
    client_db = chromadb.Client()

    try:
        client_db.delete_collection("pdf_notes_app")
    except Exception:
        pass

    collection = client_db.create_collection("pdf_notes_app")

    documents = [c["text"] for c in chunks_with_pages]
    metadatas = [
        {"page": c["page"], "source": c["source"], "chunk_index": c["chunk_index"]}
        for c in chunks_with_pages
    ]
    ids = [str(i) for i in range(len(chunks_with_pages))]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    return collection, documents, metadatas, bm25


def hybrid_search(query, collection, documents, metadatas, bm25, alpha=0.6, shortlist_size=10):
    """Combine semantic + keyword search, return a shortlist ranked by combined score."""
    semantic_results = collection.query(query_texts=[query], n_results=len(documents))
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

    tokenized_query = query.lower().split()
    bm25_raw_scores = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1
    bm25_scores = {doc: score / max_bm25 for doc, score in zip(documents, bm25_raw_scores)}

    combined = []
    for doc in documents:
        sem_score = semantic_scores.get(doc, 0)
        kw_score = bm25_scores.get(doc, 0)
        final_score = alpha * sem_score + (1 - alpha) * kw_score
        meta = doc_to_meta.get(doc, {"page": "?"})
        combined.append((doc, final_score, meta))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:shortlist_size]


def rerank(query, shortlist, top_n=3):
    """Re-rank a shortlist of (doc, score, meta) tuples using a cross-encoder."""
    docs = [item[0] for item in shortlist]
    metas = [item[2] for item in shortlist]

    pairs = [[query, doc] for doc in docs]
    scores = _reranker.predict(pairs)

    reranked = sorted(zip(docs, scores, metas), key=lambda x: x[1], reverse=True)
    return reranked[:top_n]