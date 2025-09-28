"""
PDF data loading and text embedding utilities for RAG application.

This module handles PDF document processing, text chunking, and embedding generation
using Ollama's local embedding models with parallel processing for improved performance.
"""

import concurrent.futures
import os
from typing import List
import ollama
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Configuration constants for embedding and chunking (configurable via environment)
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # Embedding dimension size
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "5"))
EMBEDDING_MAX_WORKERS = int(os.getenv("EMBEDDING_MAX_WORKERS", "4"))

# Configurable text chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Text splitter with configurable parameters for better performance
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Load a PDF file and split it into text chunks for processing.

    This function reads a PDF document, extracts text content, and splits it into
    smaller chunks suitable for embedding and retrieval.

    Args:
        path: File system path to the PDF document

    Returns:
        List of text chunks extracted from the PDF

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If PDF reading or parsing fails

    Example:
        >>> chunks = load_and_chunk_pdf("/path/to/document.pdf")
        >>> print(f"Extracted {len(chunks)} chunks")
    """
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def _embed_single_text(text: str) -> List[float]:
    """
    Generate embedding for a single text string.

    Args:
        text: Text string to embed

    Returns:
        Embedding vector as list of floats
    """
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def embed_texts(texts: List[str], progress_callback=None) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings using parallel processing.

    This function uses ThreadPoolExecutor to process multiple texts concurrently,
    significantly improving performance for large document collections.

    Args:
        texts: List of text strings to embed
        progress_callback: Optional callback function to report progress (current, total)

    Returns:
        List of embedding vectors (each vector is a list of floats)

    Raises:
        Exception: If Ollama service is unavailable or embedding fails

    Example:
        >>> texts = ["Hello world", "How are you?"]
        >>> embeddings = embed_texts(texts)
        >>> print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    """
    if not texts:
        return []

    embeddings = [None] * len(texts)  # Pre-allocate list to maintain order

    # Process in batches to control memory usage and provide progress feedback
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=EMBEDDING_MAX_WORKERS
    ) as executor:
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_indices = list(range(i, min(i + EMBEDDING_BATCH_SIZE, len(texts))))

            # Submit batch for parallel processing
            future_to_index = {
                executor.submit(_embed_single_text, text): idx
                for idx, text in zip(batch_indices, batch_texts)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    embeddings[idx] = future.result()

                    # Report progress if callback provided
                    if progress_callback:
                        completed = sum(1 for e in embeddings if e is not None)
                        progress_callback(completed, len(texts))

                except Exception as e:
                    raise Exception(
                        f"Failed to generate embedding for text at index {idx}: {e}"
                    )

    return embeddings
