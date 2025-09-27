"""
Vector database storage implementation using Qdrant for RAG application.

This module provides a wrapper around Qdrant vector database for storing and searching
document embeddings in the RAG pipeline, with connection pooling for improved performance.
"""

import os
import threading
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import Dict, List, Any, Optional

# Import embedding dimensions from data_loader module
try:
    from data_loader import EMBED_DIM
except ImportError:
    EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # Fallback if import fails

# Default similarity score threshold for search results (configurable via environment)
DEFAULT_SCORE_THRESHOLD = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.2"))


class QdrantConnectionManager:
    """
    Singleton connection manager for Qdrant client to enable connection pooling.
    
    This ensures only one Qdrant client instance is created and reused across
    all QdrantStorage instances, improving performance and reducing overhead.
    """
    _instance: Optional['QdrantConnectionManager'] = None
    _lock = threading.Lock()
    _client: Optional[QdrantClient] = None
    
    def __new__(cls) -> 'QdrantConnectionManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(QdrantConnectionManager, cls).__new__(cls)
        return cls._instance
    
    def get_client(self, url: str = None, timeout: int = 30) -> QdrantClient:
        """
        Get or create a Qdrant client instance.
        
        Args:
            url: Qdrant server URL (default from env or localhost)
            timeout: Connection timeout in seconds
            
        Returns:
            Shared QdrantClient instance
        """
        if self._client is None:
            with self._lock:
                if self._client is None:
                    if url is None:
                        url = os.getenv("QDRANT_URL", "http://localhost:6333")
                    self._client = QdrantClient(url=url, timeout=timeout)
        return self._client
    
    def close(self):
        """Close the connection and reset the client."""
        with self._lock:
            if self._client:
                self._client.close()
                self._client = None


class QdrantStorage:
    """
    Vector database storage class using Qdrant for document embeddings.
    
    This class handles the creation, storage, and retrieval of vector embeddings
    for the RAG system, providing semantic search capabilities.
    
    Attributes:
        client: Qdrant client instance
        collection: Name of the collection storing vectors
    """
    
    def __init__(self, url: str = None, collection: str = None, dim: int = None) -> None:
        """
        Initialize Qdrant storage with connection and collection setup.
        
        Uses the singleton connection manager for improved performance and resource management.
        
        Args:
            url: Qdrant server URL (default from env or localhost)
            collection: Collection name for storing vectors (default from env or "docs")
            dim: Vector dimension size (uses EMBED_DIM if None, must match embedding model)
            
        Raises:
            ConnectionError: If unable to connect to Qdrant server
        """
        # Use connection manager for pooled connections
        self._connection_manager = QdrantConnectionManager()
        self.client = self._connection_manager.get_client(url)
        
        # Get collection name from environment or use default
        if collection is None:
            collection = os.getenv("QDRANT_COLLECTION", "docs")
        self.collection = collection
        
        # Get vector dimension from parameter or use shared constant
        if dim is None:
            dim = EMBED_DIM
        self.dim = dim
        
        # Ensure collection exists with proper configuration
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Insert or update vectors in the collection with optimized batch processing.
        
        This method stores document embeddings with associated metadata for later retrieval,
        using batch processing for better performance with large datasets.
        
        Args:
            ids: List of unique identifiers for each vector
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries containing text and source info
            batch_size: Number of points to upsert in each batch (default: 100)
            
        Raises:
            Exception: If the upsert operation fails
            
        Example:
            >>> storage = QdrantStorage()
            >>> storage.upsert(
            ...     ids=["doc1_chunk1", "doc1_chunk2"],
            ...     vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            ...     payloads=[{"text": "chunk1", "source": "doc1"}, {"text": "chunk2", "source": "doc1"}]
            ... )
        """
        if not ids or not vectors or not payloads:
            return
            
        # Process in batches for better performance and memory management
        total_points = len(ids)
        for i in range(0, total_points, batch_size):
            end_idx = min(i + batch_size, total_points)
            batch_points = [
                PointStruct(id=ids[j], vector=vectors[j], payload=payloads[j]) 
                for j in range(i, end_idx)
            ]
            self.client.upsert(self.collection, points=batch_points)

    def search(self, query_vector: List[float], top_k: int = 5, score_threshold: float = None) -> Dict[str, List[str]]:
        """
        Search for similar vectors using cosine similarity with score filtering.
        
        This method finds the most relevant document chunks based on the query embedding,
        only returning results above the similarity threshold to ensure relevance.
        
        Args:
            query_vector: Embedding vector of the search query
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity score to include a result (0.0-1.0), uses env default if None
            
        Returns:
            Dictionary containing:
                - "contexts": List of text chunks from matching documents
                - "sources": List of unique source document names
                
        Raises:
            Exception: If the search operation fails
            
        Example:
            >>> storage = QdrantStorage()
            >>> results = storage.search(query_vector=[0.1, 0.2, ...], top_k=5, score_threshold=0.8)
            >>> print(f"Found {len(results['contexts'])} relevant chunks")
            >>> print(f"From sources: {results['sources']}")
        """
        if score_threshold is None:
            score_threshold = DEFAULT_SCORE_THRESHOLD
            
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
            score_threshold=score_threshold
        )
        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents stored in the vector database.
        
        Returns:
            List of dictionaries containing document information:
            - source_id: Document identifier
            - chunk_count: Number of chunks for this document
            - sample_text: Preview of document content
        
        Example:
            >>> storage = QdrantStorage()
            >>> docs = storage.list_documents()
            >>> for doc in docs:
            ...     print(f"Document: {doc['source_id']} ({doc['chunk_count']} chunks)")
        """
        try:
            # Get all points with their payloads
            scroll_result = self.client.scroll(
                collection_name=self.collection,
                with_payload=True,
                limit=10000  # Adjust based on your needs
            )
            
            points = scroll_result[0]  # First element is the list of points
            
            # Group points by source_id
            documents = {}
            for point in points:
                payload = point.payload or {}
                source_id = payload.get("source", "unknown")
                text = payload.get("text", "")
                
                if source_id not in documents:
                    documents[source_id] = {
                        "source_id": source_id,
                        "chunk_count": 0,
                        "sample_text": ""
                    }
                
                documents[source_id]["chunk_count"] += 1
                
                # Use first chunk as sample text if not set
                if not documents[source_id]["sample_text"] and text:
                    documents[source_id]["sample_text"] = text[:200] + "..." if len(text) > 200 else text
            
            return list(documents.values())
            
        except Exception as e:
            raise Exception(f"Failed to list documents: {e}")

    def delete_document(self, source_id: str) -> int:
        """
        Delete all vectors for a specific document.
        
        Args:
            source_id: The source identifier of the document to delete
            
        Returns:
            Number of vectors deleted
            
        Raises:
            Exception: If the deletion operation fails
            
        Example:
            >>> storage = QdrantStorage()
            >>> deleted_count = storage.delete_document("my_document.pdf")
            >>> print(f"Deleted {deleted_count} chunks")
        """
        try:
            # First, find all points with the given source_id
            scroll_result = self.client.scroll(
                collection_name=self.collection,
                scroll_filter={
                    "must": [
                        {
                            "key": "source",
                            "match": {"value": source_id}
                        }
                    ]
                },
                with_payload=False,  # We only need IDs
                limit=10000
            )
            
            points_to_delete = [point.id for point in scroll_result[0]]
            
            if not points_to_delete:
                return 0
            
            # Delete the points
            self.client.delete(
                collection_name=self.collection,
                points_selector=points_to_delete
            )
            
            return len(points_to_delete)
            
        except Exception as e:
            raise Exception(f"Failed to delete document '{source_id}': {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection.
        
        Returns:
            Dictionary containing collection statistics:
            - total_vectors: Total number of vectors stored
            - unique_documents: Number of unique documents
            - collection_size: Approximate collection size
            
        Example:
            >>> storage = QdrantStorage()
            >>> stats = storage.get_collection_stats()
            >>> print(f"Total vectors: {stats['total_vectors']}")
        """
        try:
            # Count unique documents
            documents = self.list_documents()
            
            # Count actual vectors by scrolling through the collection
            # This gives us the real count instead of relying on potentially incorrect metadata
            scroll_result = self.client.scroll(
                collection_name=self.collection,
                with_payload=False,  # We only need to count, not retrieve content
                limit=10000  # Should be enough for most use cases
            )
            
            actual_vector_count = len(scroll_result[0])  # Count the actual vector points
            
            # Calculate storage size estimate
            # Each vector: dimensions × 4 bytes per float + metadata overhead (~100 bytes)
            bytes_per_vector = (self.dim * 4) + 100  # Dynamic size based on actual dimensions
            total_bytes = actual_vector_count * bytes_per_vector
            
            # Format size with appropriate units
            if total_bytes < 1024:
                size_str = f"{total_bytes} B"
            elif total_bytes < 1024 * 1024:
                size_kb = total_bytes / 1024
                size_str = f"{size_kb:.1f} KB"
            elif total_bytes < 1024 * 1024 * 1024:
                size_mb = total_bytes / (1024 * 1024)
                if size_mb < 10:
                    size_str = f"{size_mb:.2f} MB"  # Show 2 decimal places for small MB values
                else:
                    size_str = f"{size_mb:.1f} MB"
            else:
                size_gb = total_bytes / (1024 * 1024 * 1024)
                size_str = f"{size_gb:.2f} GB"
            
            return {
                "total_vectors": actual_vector_count,
                "unique_documents": len(documents),
                "collection_size": size_str
            }
            
        except Exception as e:
            raise Exception(f"Failed to get collection stats: {e}")

    def search_within_document(self, query_vector: List[float], source_id: str, top_k: int = 5, score_threshold: float = None) -> Dict[str, List[str]]:
        """
        Search for similar vectors within a specific document only with score filtering.
        
        Args:
            query_vector: Embedding vector of the search query
            source_id: Limit search to this document only
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity score to include a result (0.0-1.0), uses env default if None
            
        Returns:
            Dictionary containing contexts and sources (will only contain the specified source)
            
        Example:
            >>> storage = QdrantStorage()
            >>> results = storage.search_within_document(query_vector, "document1.pdf", top_k=3, score_threshold=0.8)
        """
        try:
            if score_threshold is None:
                score_threshold = DEFAULT_SCORE_THRESHOLD
                
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter={
                    "must": [
                        {
                            "key": "source",
                            "match": {"value": source_id}
                        }
                    ]
                },
                with_payload=True,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            contexts = []
            sources = set()

            for r in results:
                payload = getattr(r, "payload", None) or {}
                text = payload.get("text", "")
                source = payload.get("source", "")
                if text:
                    contexts.append(text)
                    sources.add(source)

            return {"contexts": contexts, "sources": list(sources)}
            
        except Exception as e:
            raise Exception(f"Failed to search within document '{source_id}': {e}")