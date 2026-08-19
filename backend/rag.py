from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_PATH


COLLECTION_NAME = "aws_knowledge_v2"

# Chunk configuration
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180

# Retrieval configuration
DEFAULT_RETRIEVAL_K = 8
FETCH_K = 30


_embeddings = None
_vectorstore = None
_splitter = None


def get_embeddings():
    """
    Load the embedding model only once.

    all-MiniLM-L6-v2 is lightweight and suitable for a local
    prototype. It avoids sending document contents to an external
    embedding API.
    """
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )

    return _embeddings


def get_splitter():
    global _splitter

    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

    return _splitter


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_PATH,
        )

    return _vectorstore


def normalize_text(text: str) -> str:
    """
    Clean extracted document text while preserving paragraph structure.
    """
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Keep paragraph boundaries
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def make_document_id(
    source: str,
    page: int | None,
    chunk_index: int,
    text: str,
) -> str:
    """
    Create a deterministic ID so repeated ingestion does not
    continuously create duplicate chunks.
    """
    raw = (
        f"{source}|"
        f"{page}|"
        f"{chunk_index}|"
        f"{text}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def split_documents(
    documents: Iterable[Document],
) -> list[Document]:
    """
    Split large documents into smaller retrieval units while
    preserving source/page metadata.
    """
    splitter = get_splitter()

    chunks: list[Document] = []

    for document in documents:
        text = normalize_text(
            document.page_content
        )

        if not text:
            continue

        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page",
            None,
        )

        split_texts = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(
            split_texts
        ):
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            metadata = {
                **document.metadata,
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
                "chunk_size": len(chunk_text),
            }

            metadata["chunk_id"] = make_document_id(
                source,
                page,
                chunk_index,
                chunk_text,
            )

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )

    return chunks


def add_documents(
    documents: list[Document],
    batch_size: int = 64,
) -> int:
    """
    Split and ingest documents into ChromaDB in batches.

    Batch ingestion is important for large documents because
    embedding thousands of chunks at once can consume excessive
    memory.
    """
    if not documents:
        return 0

    chunks = split_documents(documents)

    if not chunks:
        return 0

    vectorstore = get_vectorstore()

    total = len(chunks)

    for start in range(0, total, batch_size):
        batch = chunks[
            start:start + batch_size
        ]

        ids = [
            document.metadata["chunk_id"]
            for document in batch
        ]

        vectorstore.add_documents(
            documents=batch,
            ids=ids,
        )

        print(
            f"Indexed {min(start + batch_size, total)}"
            f"/{total} chunks"
        )

    return total


def retrieve_documents(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[Document]:
    """
    MMR retrieval gives the model relevant but diverse chunks,
    reducing the chance that all retrieved chunks contain nearly
    identical text.
    """
    vectorstore = get_vectorstore()

    return vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=FETCH_K,
        lambda_mult=0.65,
    )


def retrieve_context(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> str:
    """
    Retrieve relevant chunks and format them for the LLM.
    """
    documents = retrieve_documents(
        query,
        k=k,
    )

    if not documents:
        return (
            "No relevant AWS documentation "
            "was found."
        )

    parts = []

    for index, document in enumerate(
        documents,
        1,
    ):
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )

        parts.append(
            f"SOURCE {index}\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(parts)


def get_retrieved_sources(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[dict]:
    """
    Return source metadata separately so the API/frontend can
    display citations.
    """
    documents = retrieve_documents(
        query,
        k=k,
    )

    sources = []

    for document in documents:
        sources.append(
            {
                "source": document.metadata.get(
                    "source"
                ),
                "page": document.metadata.get(
                    "page"
                ),
                "chunk_id": document.metadata.get(
                    "chunk_id"
                ),
            }
        )

    return sources