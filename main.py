"""
FastAPI backend with Inngest functions for RAG document processing.

This module provides the main API endpoints and background processing functions
for the RAG application, handling PDF ingestion and query processing using
Inngest for workflow orchestration.
"""

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import datetime
import ollama
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
  ),
)
async def rag_ingest_pdf(ctx: inngest.Context) -> dict:
    """
    Inngest function to process PDF documents for RAG ingestion.
    
    This function handles the complete PDF processing pipeline:
    1. Load and chunk the PDF document
    2. Generate embeddings for text chunks
    3. Store vectors in Qdrant database
    
    Rate limited to prevent system overload (2/min, 1 per source per 4h).
    
    Args:
        ctx: Inngest context containing event data with pdf_path and source_id
        
    Returns:
        Dictionary with ingestion results including number of chunks processed
        
    Raises:
        Exception: If PDF processing or vector storage fails
    """
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        """Load and chunk PDF from the provided path."""
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        """Generate embeddings and store vectors in Qdrant database with progress tracking."""
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        
        # Progress tracking for embeddings
        embedding_progress = {"completed": 0, "total": len(chunks)}
        
        def progress_callback(completed: int, total: int):
            embedding_progress["completed"] = completed
            embedding_progress["total"] = total
            # Note: In a real implementation, you might want to emit progress events here
            
        # Generate embeddings with progress tracking
        vecs = embed_texts(chunks, progress_callback=progress_callback)
        
        # Prepare data for vector storage
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        
        # Use shared QdrantStorage instance for better performance
        storage = QdrantStorage()
        storage.upsert(ids, vecs, payloads)
        
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> dict:
    """
    Inngest function to process user queries using RAG.
    
    This function handles the complete query processing pipeline:
    1. Embed the user question
    2. Search for relevant document chunks in Qdrant
    3. Generate answer using Ollama LLM with retrieved context
    
    Args:
        ctx: Inngest context containing event data with question and top_k
        
    Returns:
        Dictionary with answer, sources, and metadata
        
    Raises:
        Exception: If embedding, search, or LLM generation fails
    """
    def _search(question: str, top_k: int = 5, source_filter: str = None) -> RAGSearchResult:
        """Embed question and search for relevant document chunks."""
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        
        if source_filter:
            found = store.search_within_document(query_vec, source_filter, top_k)
        else:
            found = store.search(query_vec, top_k)
            
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    source_filter = ctx.event.data.get("source_filter")

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k, source_filter), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    def _generate_answer(user_content: str) -> str:
        """Generate answer using Ollama LLM with the provided context."""
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        )
        return response['message']['content'].strip()

    answer = await ctx.step.run("llm-answer", lambda: _generate_answer(user_content), output_type=str)
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI(
    title="RAG PDF Processing API",
    description="API for ingesting PDF documents and querying them using RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)


class IngestPDFRequest(BaseModel):
    """
    Request model for PDF ingestion endpoint.
    
    Attributes:
        pdf_path: Absolute file path to the PDF document
        source_id: Unique identifier for the document (typically filename)
    """
    pdf_path: str
    source_id: str


class QueryPDFRequest(BaseModel):
    """
    Request model for PDF query endpoint.
    
    Attributes:
        question: User's question about the ingested documents
        top_k: Number of relevant chunks to retrieve (default: 5)
        source_filter: Optional filter to search within specific document
    """
    question: str
    top_k: int = 5
    source_filter: str = None


class DeleteDocumentRequest(BaseModel):
    """
    Request model for document deletion endpoint.
    
    Attributes:
        source_id: Identifier of the document to delete
    """
    source_id: str


@app.post("/ingest-pdf")
async def ingest_pdf_endpoint(request: IngestPDFRequest) -> dict:
    """
    Trigger PDF document ingestion for RAG processing.
    
    This endpoint accepts a PDF document path and triggers the background
    processing pipeline to extract, embed, and store the document content.
    
    Args:
        request: IngestPDFRequest containing pdf_path and source_id
        
    Returns:
        Dictionary with event_id for tracking the ingestion process
        
    Raises:
        HTTPException: If event sending fails or service is unavailable
    """
    try:
        event_id = await inngest_client.send(inngest.Event(
            name="rag/ingest_pdf",
            data={"pdf_path": request.pdf_path, "source_id": request.source_id}
        ))
        return {"event_id": event_id[0] if event_id else "unknown"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send event: {e}")


@app.post("/query-pdf")
async def query_pdf_endpoint(request: QueryPDFRequest) -> dict:
    """
    Query ingested PDF documents using natural language.
    
    This endpoint accepts a question and triggers the RAG pipeline to find
    relevant document chunks and generate an answer using the LLM.
    
    Args:
        request: QueryPDFRequest containing question, top_k, and optional source filter
        
    Returns:
        Dictionary with event_id for tracking the query process
        
    Raises:
        HTTPException: If event sending fails or service is unavailable
    """
    try:
        event_data = {
            "question": request.question, 
            "top_k": request.top_k
        }
        if request.source_filter:
            event_data["source_filter"] = request.source_filter
            
        event_id = await inngest_client.send(inngest.Event(
            name="rag/query_pdf_ai",
            data=event_data
        ))
        return {"event_id": event_id[0] if event_id else "unknown"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send event: {e}")


@app.get("/documents")
async def list_documents_endpoint() -> dict:
    """
    List all documents stored in the vector database.
    
    Returns:
        Dictionary containing list of documents with metadata
        
    Raises:
        HTTPException: If listing documents fails
    """
    try:
        storage = QdrantStorage()
        documents = storage.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")


@app.get("/documents/stats")
async def get_collection_stats_endpoint() -> dict:
    """
    Get statistics about the document collection.
    
    Returns:
        Dictionary containing collection statistics
        
    Raises:
        HTTPException: If getting stats fails
    """
    try:
        storage = QdrantStorage()
        stats = storage.get_collection_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection stats: {e}")


@app.delete("/documents/{source_id}")
async def delete_document_endpoint(source_id: str) -> dict:
    """
    Delete a specific document from the vector database.
    
    Args:
        source_id: The identifier of the document to delete
        
    Returns:
        Dictionary with deletion results
        
    Raises:
        HTTPException: If deletion fails
    """
    try:
        storage = QdrantStorage()
        deleted_count = storage.delete_document(source_id)
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Document '{source_id}' not found")
        
        return {
            "message": f"Successfully deleted document '{source_id}'",
            "deleted_chunks": deleted_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")


@app.post("/documents/delete")
async def delete_document_post_endpoint(request: DeleteDocumentRequest) -> dict:
    """
    Delete a specific document from the vector database (POST version for Streamlit).
    
    Args:
        request: DeleteDocumentRequest containing source_id
        
    Returns:
        Dictionary with deletion results
        
    Raises:
        HTTPException: If deletion fails
    """
    return await delete_document_endpoint(request.source_id)

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])