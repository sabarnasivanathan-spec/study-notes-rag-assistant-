from groq import Groq


def get_client(api_key):
    return Groq(api_key=api_key)


def generate_answer(client, query, top_chunks):
    """top_chunks: list of (doc, score, meta) tuples"""
    context = "\n\n".join(
        f"[Page {meta['page']}]: {doc}"
        for doc, score, meta in top_chunks
    )

    prompt = f"""Answer the question based only on the context below. Mention which page(s) your answer comes from.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based only on the given context and cites page numbers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


def generate_quiz(client, sample_chunks):
    quiz_context = "\n\n".join(sample_chunks)

    quiz_prompt = f"""Based on the following content, create a 5-question multiple choice quiz.
For each question, provide 4 options (A, B, C, D) and clearly indicate the correct answer at the end.

Content:
{quiz_context}

Quiz:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates quizzes from study material."},
            {"role": "user", "content": quiz_prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content