
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")


def create_rag_components():
    """Load ChromaDB and the Gemini chat model."""

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or "PASTE_YOUR" in api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Check your .env file."
        )

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "ChromaDB is missing. Run "
            "python build_knowledge_base.py first."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    vectorstore = Chroma(
        collection_name="industrial_maintenance",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0.2,
    )

    return vectorstore, llm


def answer_with_rag(question, machine_context, vectorstore, llm):
    """Retrieve relevant passages and generate a readable answer."""

    retrieval_query = (
        f"Question: {question}\n"
        f"Machine assessment: {machine_context}"
    )

    documents = vectorstore.similarity_search(
        retrieval_query,
        k=4,
    )

    if documents:
        context_text = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n"
            f"{doc.page_content}"
            for doc in documents
        )
    else:
        context_text = (
            "No relevant knowledge-base passages were retrieved. "
            "Clearly state this limitation in your answer."
        )

    prompt = f"""
You are an industrial maintenance decision-support assistant.

Answer the user's question using the available evidence.

Rules:
- Do not invent machine specifications or safe operating limits.
- The failure-risk score may not be a calibrated probability.
- Anomaly detection does not prove a failure or model error.
- Never claim that a machine is safe based on a low model score.
- Explain uncertainty and recommend appropriate verification.
- Follow the machine manual and approved site procedures.
- Never recommend bypassing safety guards or interlocks.
- Cite source filenames only when supported by retrieved passages.
- If evidence is missing, say so clearly.

USER QUESTION:
{question}

CURRENT MACHINE ASSESSMENT:
{machine_context}

RETRIEVED KNOWLEDGE:
{context_text}

Use these headings:
### 1. Interpretation
### 2. Relevant evidence
### 3. Recommended checks
### 4. Limitations and uncertainty
"""

    response = llm.invoke(prompt)
    content = response.content

    # Convert Gemini's response into plain text.
    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text_part = item.get("text")
                if text_part:
                    parts.append(str(text_part))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))

        content = "\n".join(parts)

    clean_answer = str(content)

    return {
        "answer": clean_answer,
        "documents": documents,
    }
