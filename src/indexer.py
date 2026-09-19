import os
import json
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Persistence directory and model selection
DB_DIR = "./chroma_db"
# Local embedding model as required by ground rules (no hosted embedding APIs)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(file_path: str = "corpus.jsonl") -> List[Document]:
    """
    Reads corpus.jsonl line-by-line and converts each chunk into a LangChain Document.
    Preserves all metadata required for strict citation verification and recency resolution.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Corpus file '{file_path}' not found at root directory. "
            "Make sure 'corpus.jsonl' is placed in the project root."
        )

    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue

            # Metadata dictionary mapping exact corpus fields
            metadata = {
                "chunk_id": item.get("chunk_id", ""),
                "doc_id": item.get("doc_id", ""),
                "title": item.get("title", ""),
                "category": item.get("category", ""),
                "owner": item.get("owner", ""),
                "published": item.get("published", ""),  # Crucial for resolving recency conflicts
                "version": str(item.get("version", "")),
                "source_url": item.get("source_url", "")
            }

            doc = Document(
                page_content=item.get("text", ""),
                metadata=metadata
            )
            documents.append(doc)

    print(f"Loaded {len(documents)} chunks from '{file_path}'.")
    return documents


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Loads local SentenceTransformers model on CPU/GPU without external API calls.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vector_store(corpus_path: str = "corpus.jsonl") -> Chroma:
    """
    Loads chunks, embeds them locally, and builds or replaces the persistent Chroma collection.
    """
    print("Initializing local HuggingFace embeddings...")
    embeddings = get_embedding_function()

    print("Parsing corpus.jsonl...")
    docs = load_corpus(corpus_path)

    print(f"Building Chroma vector database at '{DB_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"Successfully indexed {len(docs)} chunks into ChromaDB at '{DB_DIR}'.")
    return vectorstore


def get_vector_store() -> Chroma:
    """
    Retrieves the existing persistent Chroma vector store.
    Auto-indexes if the DB folder does not exist yet.
    """
    embeddings = get_embedding_function()
    if not os.path.exists(DB_DIR):
        print(f"Database directory '{DB_DIR}' not found. Building index now...")
        return build_vector_store()
    
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )


if __name__ == "__main__":
    # Test/standalone execution to build the database
    build_vector_store()