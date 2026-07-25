import chromadb

client = chromadb.Client()
collection = client.create_collection("test")

collection.add(
    documents=["The sky is blue", "Bananas are yellow", "The ocean is blue too"],
    ids=["1", "2", "3"]
)

results = collection.query(
    query_texts=["What color is the sky?"],
    n_results=2
)

print(results)