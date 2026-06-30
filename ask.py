import requests
import chromadb

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pdf_collection"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b"


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        }
    )

    response.raise_for_status()
    return response.json()["embedding"]


def ask_ollama(question, context):
    prompt = f"""
You are a PDF question-answering assistant.

Answer ONLY using the provided PDF context.

If the answer is not found in the context, say:
"I could not find this information in the PDF."

PDF context:
{context}

Question:
{question}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()
    return response.json()["response"]


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    question = input("Ask a question about the PDF: ")

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    retrieved_chunks = results["documents"][0]

    context = "\n\n".join(retrieved_chunks)

    answer = ask_ollama(question, context)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()