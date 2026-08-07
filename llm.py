from groq import Groq


def get_client(api_key):
    return Groq(api_key=api_key)


def generate_answer(client, query, top_chunks, chat_history=None):
    """top_chunks: list of (doc, score, meta) tuples
    chat_history: list of {"question": ..., "answer": ...} dicts, most recent last
    """
    context = "\n\n".join(
        f"[Page {meta['page']}]: {doc}"
        for doc, score, meta in top_chunks
    )

    history_text = ""
    if chat_history:
        history_text = "\n\n".join(
            f"Q: {turn['question']}\nA: {turn['answer']}"
            for turn in chat_history[-3:]
        )
        history_text = f"Previous conversation:\n{history_text}\n\n"

    prompt = f"""Answer the question based only on the context below. Mention which page(s) your answer comes from.
Use the previous conversation (if any) to understand follow-up questions, but only answer using facts from the context.

{history_text}Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based only on the given context and cites page numbers. You can use conversation history to understand follow-up questions."},
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