from groq import Groq

client = Groq(api_key="YOUR_API_KEY_HERE")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain photosynthesis in one sentence."}
    ],
    temperature=0.1
)

print(response.choices[0].message.content)