"""  
Streamlit frontend for RAG PDF processing application.

This module provides a user-friendly web interface for uploading PDF documents
and querying them using the RAG (Retrieval-Augmented Generation) system.
It communicates with the FastAPI backend and monitors Inngest workflow execution.
"""

from pathlib import Path
import time
import streamlit as st
from dotenv import load_dotenv
import os
import requests
from typing import Tuple, Optional

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


def _inngest_api_base() -> str:
    """Get the Inngest API base URL from environment variables."""
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def _get_timeout_config() -> Tuple[float, float]:
    """
    Get timeout configuration from environment variables.
    
    Returns:
        Tuple of (timeout_seconds, poll_interval_seconds)
    """
    timeout_s = float(os.getenv("QUERY_TIMEOUT_SECONDS", "300"))
    poll_interval_s = float(os.getenv("POLL_INTERVAL_SECONDS", "0.5"))
    return timeout_s, poll_interval_s


def _should_auto_cleanup() -> bool:
    """
    Check if automatic file cleanup is enabled.
    
    Returns:
        True if uploaded files should be automatically deleted after processing
    """
    return os.getenv("AUTO_CLEANUP_UPLOADS", "true").lower() == "true"


def _make_api_request(endpoint: str, json_data: dict = None, method: str = "POST") -> Tuple[Optional[dict], int]:
    """
    Make an API request to the FastAPI backend with consistent error handling.
    
    Args:
        endpoint: API endpoint path (e.g., "/ingest-pdf")
        json_data: JSON data to send in request body
        method: HTTP method (GET or POST)
        
    Returns:
        Tuple of (response JSON data or None, status code)
    """
    try:
        url = f"http://127.0.0.1:8000{endpoint}"
        if method.upper() == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=json_data)
        
        if response.status_code == 200:
            return response.json(), response.status_code
        else:
            st.error(f"API request failed with status {response.status_code}")
            return None, response.status_code
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None, 0


def _keep_files_on_error() -> bool:
    """
    Check if files should be kept when processing fails.
    
    Returns:
        True if files should be preserved for debugging when errors occur
    """
    return os.getenv("KEEP_FILES_ON_ERROR", "true").lower() == "true"


# Display configuration info
timeout_s, poll_interval_s = _get_timeout_config()
cleanup_enabled = _should_auto_cleanup()

with st.expander("⚙️ Configuration", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Timeout & Cleanup**")
        st.write(f"• Query Timeout: {timeout_s}s")
        st.write(f"• Auto Cleanup: {'✅ Enabled' if cleanup_enabled else '❌ Disabled'}")
        st.write(f"• Keep Files on Error: {'✅ Yes' if _keep_files_on_error() else '❌ No'}")
    
    with col2:
        st.write("**Performance Settings**")
        embedding_model = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
        batch_size = os.getenv("EMBEDDING_BATCH_SIZE", "5")
        max_workers = os.getenv("EMBEDDING_MAX_WORKERS", "4")
        chunk_size = os.getenv("CHUNK_SIZE", "1500")
        
        st.write(f"• Embedding Model: {embedding_model.split(':')[0]}")
        st.write(f"• Parallel Workers: {max_workers}")
        st.write(f"• Batch Size: {batch_size}")
        st.write(f"• Chunk Size: {chunk_size} chars")
    
    st.caption("💡 Configure these settings in the `.env` file")


def save_uploaded_pdf(file) -> Path:
    """
    Save an uploaded PDF file to the local uploads directory.
    
    Args:
        file: Streamlit UploadedFile object containing the PDF
        
    Returns:
        Path object pointing to the saved file
        
    Raises:
        OSError: If file writing fails due to permissions or disk space
    """
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


def cleanup_uploaded_file(file_path: Path, success: bool = True) -> None:
    """
    Clean up uploaded file based on configuration and processing success.
    
    This function implements the file cleanup policy based on environment
    configuration and whether the processing was successful.
    
    Args:
        file_path: Path to the uploaded file
        success: Whether the processing was successful
        
    Note:
        Files are kept for debugging if processing fails and KEEP_FILES_ON_ERROR=true
    """
    try:
        if not file_path.exists():
            return
            
        should_cleanup = _should_auto_cleanup()
        keep_on_error = _keep_files_on_error()
        
        # Clean up if auto cleanup is enabled and either succeeded or failed but not keeping on error
        if should_cleanup and (success or not keep_on_error):
            file_path.unlink()
            st.caption(f"🗑️ Cleaned up temporary file: {file_path.name}")
        elif not success and keep_on_error:
            st.caption(f"📁 Keeping file for debugging: {file_path.name}")
    except Exception as e:
        st.warning(f"Failed to cleanup file {file_path.name}: {e}")


def cleanup_orphaned_files() -> None:
    """
    Clean up any leftover files from previous application sessions.
    
    This function runs on application startup to remove any PDF files
    that were left behind from previous sessions, helping maintain clean storage.
    
    Note:
        Only runs if AUTO_CLEANUP_UPLOADS is enabled in configuration
    """
    try:
        uploads_dir = Path("uploads")
        if not uploads_dir.exists():
            return
            
        # Only clean up if auto cleanup is enabled
        if not _should_auto_cleanup():
            return
            
        files = list(uploads_dir.glob("*.pdf"))
        if files:
            for file_path in files:
                file_path.unlink()
            st.caption(f"🧹 Cleaned up {len(files)} orphaned files from previous sessions")
    except Exception as e:
        st.warning(f"Failed to cleanup orphaned files: {e}")


def send_ingest_request(pdf_path: Path) -> Optional[str]:
    """
    Send PDF ingestion request to the FastAPI backend.
    
    This function triggers the background PDF processing pipeline by sending
    a request to the FastAPI ingestion endpoint.
    
    Args:
        pdf_path: Path to the uploaded PDF file
        
    Returns:
        Event ID for tracking the ingestion process, or None if request failed
    """
    result, status = _make_api_request("/ingest-pdf", {
        "pdf_path": str(pdf_path.resolve()),
        "source_id": pdf_path.name
    })
    return result.get("event_id", "unknown") if result else None


def send_query_request(question: str, top_k: int, source_filter: str = None) -> Optional[str]:
    """
    Send query request to the FastAPI backend.
    
    This function triggers the RAG query processing pipeline by sending
    a user question to the FastAPI query endpoint.
    
    Args:
        question: User's natural language question
        top_k: Number of relevant document chunks to retrieve
        source_filter: Optional filter to search within specific document
        
    Returns:
        Event ID for tracking the query process, or None if request failed
    """
    request_data = {
        "question": question,
        "top_k": top_k
    }
    if source_filter:
        request_data["source_filter"] = source_filter
        
    result, status = _make_api_request("/query-pdf", request_data)
    return result.get("event_id", "unknown") if result else None


def fetch_runs(event_id: str) -> list[dict]:
    """
    Fetch execution runs for a given Inngest event ID.
    
    This function queries the Inngest API to get the status and results
    of background workflow executions.
    
    Args:
        event_id: Unique identifier for the Inngest event
        
    Returns:
        List of run dictionaries containing status and output information
        
    Raises:
        requests.RequestException: If the API request fails
    """
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = None, poll_interval_s: float = None, progress_placeholder=None, operation_type: str = "ingestion") -> dict:
    """
    Wait for Inngest workflow completion and return the results.
    
    This function polls the Inngest API until the workflow completes or times out,
    providing real-time progress updates to the user interface.
    
    Args:
        event_id: Unique identifier for the Inngest event to monitor
        timeout_s: Maximum time to wait in seconds (from config if None)
        poll_interval_s: Polling frequency in seconds (from config if None) 
        progress_placeholder: Streamlit element for displaying progress updates
        operation_type: Type of operation ('ingestion' or 'query') for appropriate messages
        
    Returns:
        Dictionary containing the workflow output/results
        
    Raises:
        TimeoutError: If the workflow doesn't complete within the timeout
        RuntimeError: If the workflow fails or is cancelled
    """
    if timeout_s is None or poll_interval_s is None:
        timeout_s, poll_interval_s = _get_timeout_config()
    
    start = time.time()
    last_status = None
    
    # Define operation-specific messages
    if operation_type == "query":
        status_messages = {
            "Running": "🔍 Processing query...",
            "Queued": "⏳ Query queued, waiting to start...",
            "Started": "🚀 Starting query processing...",
        }
        step_messages = {
            "load-and-chunk": "📄 Loading document context...",
            "embed-and-upsert": "🧠 Processing embeddings...",
            "embed-and-search": "🔍 Searching for relevant content...",
            "llm-answer": "✨ Generating answer with AI...",
        }
    else:  # ingestion
        status_messages = {
            "Running": "🔄 Processing document...",
            "Queued": "⏳ Request queued, waiting to start...",
            "Started": "🚀 Starting processing...",
        }
        step_messages = {
            "load-and-chunk": "📄 Loading and chunking PDF document...",
            "embed-and-upsert": "🧠 Generating embeddings and storing vectors...",
            "embed-and-search": "🔍 Searching for relevant content...",
            "llm-answer": "✨ Generating answer with AI...",
        }
    
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            
            # Update progress if placeholder provided
            if progress_placeholder and status:
                elapsed = time.time() - start
                message = status_messages.get(status, f"📊 Status: {status}")
                
                # Try to get more detailed step information
                step_info = ""
                if run.get("steps"):
                    current_step = None
                    for step in run["steps"]:
                        if step.get("status") in ("Running", "Started"):
                            current_step = step.get("name", "")
                            break
                    if current_step and current_step in step_messages:
                        step_info = f" - {step_messages[current_step]}"
                
                progress_placeholder.info(f"{message}{step_info} (Elapsed: {elapsed:.1f}s)")
            
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                if progress_placeholder:
                    progress_placeholder.success("✅ Processing completed successfully!")
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                error_msg = f"Function run {status}"
                if run.get("output") and run["output"].get("error"):
                    error_msg += f": {run['output']['error']}"
                raise RuntimeError(error_msg)
        
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(f"Timed out after {timeout_s}s waiting for run output (last status: {last_status}). Try increasing QUERY_TIMEOUT_SECONDS in .env file.")
        time.sleep(poll_interval_s)


# Cleanup orphaned files on app start
cleanup_orphaned_files()

# Main application with tabs
st.title("🤖 RAG Document Manager")
st.caption("Upload, manage, and query your PDF documents with AI")

# Create tabs
tab_upload, tab_query, tab_manage = st.tabs(["📄 Upload", "🔍 Query", "📚 Manage"])

with tab_upload:
    st.header("Upload a PDF to Ingest")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

    # Initialize session state for upload tracking
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "currently_processing" not in st.session_state:
        st.session_state.currently_processing = None

    # Generate a unique identifier for the uploaded file
    file_id = None
    if uploaded is not None:
        # Create file identifier based on name and size
        file_id = f"{uploaded.name}_{uploaded.size}"
        
        # Show file info
        st.info(f"📄 Selected: {uploaded.name} ({uploaded.size} bytes)")
        
        # Check if this file has already been processed
        if file_id in st.session_state.processed_files:
            st.success(f"✅ {uploaded.name} has already been processed and ingested!")
            if st.button("🔄 Process Again", help="Re-process this file"):
                st.session_state.processed_files.discard(file_id)
                st.rerun()
        # Check if this file is currently being processed
        elif st.session_state.currently_processing == file_id:
            st.warning("⏳ This file is currently being processed. Please wait...")
        # Process new file
        else:
            if st.button(f"🚀 Upload and Process {uploaded.name}", type="primary"):
                st.session_state.currently_processing = file_id
    # Process file only if marked for processing
    if uploaded is not None and st.session_state.currently_processing == file_id:
        path = None
        try:
            with st.spinner("Uploading and triggering ingestion..."):
                path = save_uploaded_pdf(uploaded)
                # Send event to FastAPI
                event_id = send_ingest_request(path)
                
                if event_id:
                    st.success(f"✅ Triggered ingestion for: {path.name}")
                    st.info("📊 Processing document in background...")
                    
                    # Wait for processing to complete with progress tracking
                    progress_placeholder = st.empty()
                    try:
                        timeout_s, _ = _get_timeout_config()
                        result = wait_for_run_output(event_id, progress_placeholder=progress_placeholder, operation_type="ingestion")
                        
                        if result and result.get("ingested", 0) > 0:
                            st.success(f"🎉 Successfully ingested {result['ingested']} chunks from {path.name}")
                            cleanup_uploaded_file(path, success=True)
                            
                            # Mark file as successfully processed
                            st.session_state.processed_files.add(file_id)
                            st.session_state.currently_processing = None
                            
                            st.info("💡 You can now query this document in the Query tab, or upload another PDF here.")
                        else:
                            st.warning("⚠️ Ingestion completed but no chunks were processed")
                            cleanup_uploaded_file(path, success=False)
                            st.session_state.currently_processing = None
                            
                    except (RuntimeError, TimeoutError) as e:
                        st.error(f"❌ Ingestion failed: {e}")
                        cleanup_uploaded_file(path, success=False)
                        st.session_state.currently_processing = None
                    finally:
                        progress_placeholder.empty()
                        
                else:
                    st.error("❌ Failed to trigger ingestion. Please check if all services are running.")
                    if path:
                        cleanup_uploaded_file(path, success=False)
                    st.session_state.currently_processing = None
                        
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")
            if path:
                cleanup_uploaded_file(path, success=False)
            st.session_state.currently_processing = None

with tab_query:
    st.header("Ask a question about your PDFs")
    
    # Query Section
    with st.form("rag_query_form"):
        question = st.text_input("Your question")
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
        with col2:
            # Get list of documents for filtering
            docs_data, status = _make_api_request("/documents", method="GET")
            if status == 200 and docs_data:
                doc_options = ["All documents"] + [doc["source_id"] for doc in docs_data.get("documents", [])]
                source_filter = st.selectbox("Search in specific document", doc_options)
                if source_filter == "All documents":
                    source_filter = None
            else:
                source_filter = None
                st.caption("⚠️ Could not load document list")
        
        submitted = st.form_submit_button("Ask")

        if submitted and question.strip():
            with st.spinner("Sending query request..."):
                # Send query event to FastAPI
                event_id = send_query_request(question.strip(), int(top_k), source_filter)
                
                if event_id:
                    st.info("🔍 Searching documents and generating answer...")
                    
                    # Create progress placeholder
                    progress_placeholder = st.empty()
                    
                    try:
                        # Poll the local Inngest API for the run's output with progress tracking
                        output = wait_for_run_output(event_id, progress_placeholder=progress_placeholder, operation_type="query")
                        answer = output.get("answer", "")
                        sources = output.get("sources", [])
                        num_contexts = output.get("num_contexts", 0)

                        # Display results
                        st.subheader("📝 Answer")
                        if answer:
                            st.write(answer)
                            st.caption(f"💡 Generated from {num_contexts} document chunks")
                        else:
                            st.warning("⚠️ No answer could be generated from the available documents")
                        
                        if sources:
                            st.subheader("📚 Sources")
                            for i, source in enumerate(sources, 1):
                                st.write(f"{i}. {source}")
                        else:
                            st.info("ℹ️ No sources found for this query")
                            
                    except (RuntimeError, TimeoutError) as e:
                        st.error(f"❌ Query failed: {e}")
                        if "Timed out" in str(e):
                            timeout_s, _ = _get_timeout_config()
                            st.info(f"💡 Current timeout: {timeout_s}s. You can increase QUERY_TIMEOUT_SECONDS in the .env file for longer queries.")
                    finally:
                        progress_placeholder.empty()
                else:
                    st.error("❌ Failed to send query. Please check if all services are running.")

with tab_manage:
    st.header("Document Management")
    
    # Show collection statistics
    try:
        stats_data, status = _make_api_request("/documents/stats", method="GET")
        if status == 200 and stats_data:
            stats = stats_data.get("stats", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Vectors", stats.get("total_vectors", 0))
            with col2:
                st.metric("Documents", stats.get("unique_documents", 0))
            with col3:
                st.metric("Storage Size", stats.get("collection_size", "0 MB"))
        else:
            st.error("Failed to load collection statistics")
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")
    
    st.divider()
    
    # Document list and management
    try:
        docs_data, status = _make_api_request("/documents", method="GET")
        if status == 200 and docs_data:
            documents = docs_data.get("documents", [])
            
            if documents:
                st.subheader("📄 Your Documents")
                
                for doc in documents:
                    with st.expander(f"📄 {doc['source_id']}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Chunks:** {doc['chunk_count']}")
                            preview_text = doc.get('sample_text', 'No preview available')
                            st.write(f"**Preview:** {preview_text[:150]}..." if len(preview_text) > 150 else f"**Preview:** {preview_text}")
                        
                        with col2:
                            # Use session state to track deletion confirmation
                            confirm_key = f"confirm_delete_{doc['source_id']}"
                            
                            if st.button("🗑️ Delete", key=f"delete_{doc['source_id']}", type="secondary"):
                                st.session_state[confirm_key] = True
                            
                            # Show confirmation if delete was clicked
                            if st.session_state.get(confirm_key, False):
                                st.warning(f"⚠️ Delete {doc['source_id']}?")
                                col_yes, col_no = st.columns(2)
                                
                                with col_yes:
                                    if st.button("✅ Yes", key=f"yes_{doc['source_id']}", type="primary"):
                                        result, status = _make_api_request("/documents/delete", {"source_id": doc['source_id']})
                                        if status == 200 and result:
                                            st.success(f"✅ {result['message']}")
                                            del st.session_state[confirm_key]  # Clear confirmation state
                                            st.rerun()  # Refresh the page
                                        else:
                                            st.error("Failed to delete document")
                                            if confirm_key in st.session_state:
                                                del st.session_state[confirm_key]
                                
                                with col_no:
                                    if st.button("❌ No", key=f"no_{doc['source_id']}", type="secondary"):
                                        del st.session_state[confirm_key]
            else:
                st.info("📭 No documents uploaded yet. Go to the Upload tab to add some!")
        else:
            st.error("Failed to load documents")
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")
    
    # Upload state management
    if st.button("🗑️ Clear Upload History", help="Clear the list of processed files"):
        st.session_state.processed_files = set()
        st.session_state.currently_processing = None
        st.success("✅ Upload history cleared!")
        st.rerun()
    
    # Show processed files info
    if st.session_state.processed_files:
        with st.expander(f"📋 Processed Files ({len(st.session_state.processed_files)})", expanded=False):
            for file_id in st.session_state.processed_files:
                st.caption(f"✅ {file_id.split('_')[0]}")
    
    # Bulk operations
    st.divider()
    
    if documents:  # Only show bulk operations if there are documents
        bulk_confirm_key = "confirm_bulk_delete"
        
        if st.button("🗑️ Clear All Documents", type="secondary"):
            st.session_state[bulk_confirm_key] = True
        
        if st.session_state.get(bulk_confirm_key, False):
            st.warning("⚠️ This will delete ALL documents from your knowledge base!")
            col_yes, col_no = st.columns(2)
            
            with col_yes:
                if st.button("🚨 DELETE ALL", type="primary"):
                    deleted_count = 0
                    errors = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, doc in enumerate(documents):
                        status_text.text(f"Deleting {doc['source_id']}...")
                        result, status = _make_api_request("/documents/delete", {"source_id": doc['source_id']})
                        if status == 200:
                            deleted_count += 1
                        else:
                            errors.append(doc['source_id'])
                        
                        progress_bar.progress((i + 1) / len(documents))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if deleted_count > 0:
                        st.success(f"✅ Successfully deleted {deleted_count} documents")
                    if errors:
                        st.error(f"❌ Failed to delete: {', '.join(errors)}")
                    
                    del st.session_state[bulk_confirm_key]
                    st.rerun()
            
            with col_no:
                if st.button("❌ Cancel", type="secondary"):
                    del st.session_state[bulk_confirm_key]