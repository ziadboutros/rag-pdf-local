import requests
import chromadb
from pypdf import PdfReader

PDF_PATH = "data/sample.pdf"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pdf_collection"


def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )

    response.raise_for_status()
    return response.json()["embedding"]


def main():
    print("Reading PDF...")
    text = read_pdf(PDF_PATH)

    # print("Splitting text into chunks...")
    # chunks = split_text(text)

    # print(f"Total chunks created: {len(chunks)}")
    print("Splitting text into chunks...")
    chunks = split_text(text)

    print(f"Total chunks created: {len(chunks)}")

    print("\nShowing chunks separately:")
    print("=" * 100)

    for i, chunk in enumerate(chunks):
        print(f"\n\n========== CHUNK {i} ==========")
        print(chunk)
        print("========== END OF CHUNK ==========\n")

    return

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        print(f"Creating embedding for chunk {i + 1}/{len(chunks)}")

        embedding = get_embedding(chunk)

        ids.append(f"chunk_{i}")
        documents.append(chunk)
        embeddings.append(embedding)
        metadatas.append({
            "source": PDF_PATH,
            "chunk_number": i
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print("PDF saved successfully into ChromaDB.")


if __name__ == "__main__":
    main()
