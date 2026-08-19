from __future__ import annotations

import re
import sys
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


# ============================================================
# PROJECT PATH
# ============================================================

# Project root:
# aws-ai-agent/
#
# scripts/
#     ingest_rag.py
#
# rag_data/
#     document1.pdf
#     document2.pdf
#     document3.pdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAG_DIRECTORY = PROJECT_ROOT / "rag_data"


# ============================================================
# IMPORT BACKEND RAG MODULE
# ============================================================

# Make sure Python can find the backend package.
sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from backend.rag import add_documents


# ============================================================
# CONFIGURATION
# ============================================================

# Supported document types.
SUPPORTED_EXTENSIONS = {
    ".pdf",
}

# Number of source pages processed before printing progress.
PROGRESS_INTERVAL = 100

# Chroma embedding batch size.
# Keep this moderate because your documents are very large.
EMBEDDING_BATCH_SIZE = 64


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving useful
    paragraph and line boundaries.
    """

    if not text:
        return ""

    # Remove null characters.
    text = text.replace("\x00", " ")

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces/tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# LOAD ONE PDF
# ============================================================

def load_pdf(pdf_path: Path) -> list[Document]:
    """
    Load one PDF page-by-page using PyMuPDF.

    Each PDF page becomes a LangChain Document.

    Metadata contains:
        source
        file_path
        page
        document_name
    """

    print()
    print("=" * 70)
    print(f"Loading PDF: {pdf_path.name}")
    print("=" * 70)

    try:
        loader = PyMuPDFLoader(
            str(pdf_path)
        )

        pages = loader.load()

    except Exception as exc:
        print(
            f"ERROR loading {pdf_path.name}: {exc}"
        )

        return []

    documents: list[Document] = []

    total_pages = len(pages)

    print(
        f"Pages detected: {total_pages}"
    )

    for index, page_document in enumerate(
        pages
    ):

        text = clean_text(
            page_document.page_content
        )

        # Ignore completely empty pages.
        if not text:
            continue

        # PyMuPDF/LangChain normally uses
        # zero-based page metadata.
        #
        # We convert it to human-friendly
        # one-based page numbering.
        original_page = page_document.metadata.get(
            "page"
        )

        if original_page is not None:
            page_number = int(
                original_page
            ) + 1
        else:
            page_number = index + 1

        metadata = {
            "source": pdf_path.name,
            "file_path": str(pdf_path),
            "document_name": pdf_path.stem,
            "page": page_number,
            "file_type": "pdf",
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

        if (
            (index + 1) % PROGRESS_INTERVAL == 0
            or index + 1 == total_pages
        ):
            print(
                f"  Processed pages: "
                f"{index + 1}/{total_pages}"
            )

    print(
        f"Usable pages extracted: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# LOAD ALL PDFs
# ============================================================

def load_all_pdfs() -> list[Document]:
    """
    Find and load every PDF inside rag_data/.
    """

    if not RAG_DIRECTORY.exists():
        print()
        print(
            f"ERROR: RAG directory does not exist:"
        )
        print(
            f"       {RAG_DIRECTORY}"
        )
        print()
        return []

    pdf_files = sorted(
        [
            path
            for path in RAG_DIRECTORY.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    if not pdf_files:
        print()
        print(
            "ERROR: No PDF files found in:"
        )
        print(
            f"       {RAG_DIRECTORY}"
        )
        print()
        print(
            "Place your PDFs inside the rag_data folder."
        )
        return []

    print()
    print("=" * 70)
    print("RAG DOCUMENT INGESTION")
    print("=" * 70)

    print(
        f"RAG directory: {RAG_DIRECTORY}"
    )

    print(
        f"PDF files found: {len(pdf_files)}"
    )

    for index, pdf in enumerate(
        pdf_files,
        start=1
    ):
        print(
            f"  {index}. {pdf.name}"
        )

    print()

    all_documents: list[Document] = []

    for pdf_path in pdf_files:

        documents = load_pdf(
            pdf_path
        )

        all_documents.extend(
            documents
        )

    return all_documents


# ============================================================
# DISPLAY DOCUMENT STATISTICS
# ============================================================

def display_statistics(
    documents: list[Document],
) -> None:
    """
    Display basic statistics about the source documents.
    """

    print()
    print("=" * 70)
    print("DOCUMENT STATISTICS")
    print("=" * 70)

    print(
        f"Total usable pages: {len(documents)}"
    )

    documents_by_source: dict[str, int] = {}

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown"
        )

        documents_by_source[source] = (
            documents_by_source.get(
                source,
                0
            )
            + 1
        )

    for source, count in (
        documents_by_source.items()
    ):
        print(
            f"  {source}: {count} pages"
        )


# ============================================================
# INGEST INTO CHROMA
# ============================================================

def ingest_documents(
    documents: list[Document],
) -> None:
    """
    Send source documents to backend.rag.

    backend.rag is responsible for:
        - chunking
        - creating embeddings
        - storing vectors in ChromaDB
    """

    print()
    print("=" * 70)
    print("STARTING RAG INDEXING")
    print("=" * 70)

    print()
    print(
        "The documents will now be:"
    )

    print(
        "  1. Split into chunks"
    )

    print(
        "  2. Converted into embeddings"
    )

    print(
        "  3. Stored in ChromaDB"
    )

    print()
    print(
        "This may take a while for ~18,000 pages."
    )

    print()

    total_chunks = add_documents(
        documents,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    print()
    print("=" * 70)
    print("RAG INDEXING COMPLETE")
    print("=" * 70)

    print(
        f"Source pages: {len(documents)}"
    )

    print(
        f"Chunks indexed: {total_chunks}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("AWS MONITORING AGENT - RAG INGESTION")
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Knowledge directory: {RAG_DIRECTORY}"
    )

    # --------------------------------------------------------
    # STEP 1: Load PDFs
    # --------------------------------------------------------

    documents = load_all_pdfs()

    if not documents:

        print()
        print(
            "No documents were loaded."
        )

        return

    # --------------------------------------------------------
    # STEP 2: Display statistics
    # --------------------------------------------------------

    display_statistics(
        documents
    )

    # --------------------------------------------------------
    # STEP 3: Ingest into ChromaDB
    # --------------------------------------------------------

    ingest_documents(
        documents
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL DONE")
    print("=" * 70)

    print()
    print(
        "Your documents are now available in ChromaDB."
    )

    print()
    print(
        "You can now start the FastAPI backend and ask"
    )

    print(
        "knowledge questions through the chatbot."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    