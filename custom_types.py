"""
Pydantic models for RAG application type definitions.

This module defines data structures used throughout the RAG (Retrieval-Augmented Generation)
application for type safety and data validation.
"""

import pydantic
from typing import Optional


class RAGChunkAndSrc(pydantic.BaseModel):
    """
    Represents PDF chunks and their source information.
    
    Attributes:
        chunks: List of text chunks extracted from the PDF
        source_id: Identifier for the source document (e.g., filename)
    """
    chunks: list[str]
    source_id: Optional[str] = None


class RAGUpsertResult(pydantic.BaseModel):
    """
    Result of upserting documents into the vector database.
    
    Attributes:
        ingested: Number of chunks successfully ingested
    """
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    """
    Result of searching the vector database for relevant contexts.
    
    Attributes:
        contexts: List of relevant text chunks found
        sources: List of unique source document identifiers
    """
    contexts: list[str]
    sources: list[str]


