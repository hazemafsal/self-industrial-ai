
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = (
    BASE_DIR / "knowledge_base" / "industrial_maintenance.txt"
)
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv()


def main():
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_FILE}"
        )

    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": KNOWLEDGE_FILE.name}],
    )

    if not chunks:
        raise ValueError("No knowledge chunks were created.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # Remove chroma_db before rebuilding if you want a clean re-index.
    # Stop Streamlit first if it is using this database.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="industrial_maintenance",
    )

    print("Knowledge base indexed successfully.")
    print(f"Source file: {KNOWLEDGE_FILE}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"ChromaDB directory: {CHROMA_DIR}")
    print(f"Collection: {vectorstore._collection.name}")


if __name__ == "__main__":
    if not __import__("os").environ.get("GOOGLE_API_KEY"):
        # load_dotenv() is called above, so the environment is checked
        # after loading the .env file.
        pass
    main()
