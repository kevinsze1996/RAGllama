#!/bin/bash

# RAG Application Startup Script
# This script starts all components of your local RAG application with robust error handling

set -e  # Exit on any error

# Configuration
INNGEST_PORT=8288
QDRANT_PORT=6333
FASTAPI_PORT=8000
STREAMLIT_PORT=8502

# Trap to cleanup processes on script exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [[ -n "$INNGEST_PID" ]]; then kill $INNGEST_PID 2>/dev/null || true; fi
    if [[ -n "$QDRANT_PID" ]]; then kill $QDRANT_PID 2>/dev/null || true; fi
    if [[ -n "$FASTAPI_PID" ]]; then kill $FASTAPI_PID 2>/dev/null || true; fi
    
    # Kill any remaining processes by port
    for port in $INNGEST_PORT $QDRANT_PORT $FASTAPI_PORT $STREAMLIT_PORT; do
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
    done
    
    echo "✅ All services stopped"
}
trap cleanup EXIT INT TERM

# Function to check if port is available
check_port() {
    local port=$1
    local service=$2
    if lsof -i :$port >/dev/null 2>&1; then
        echo "⚠️  Port $port is already in use by another process"
        echo "   Killing existing process on port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 1
        if lsof -i :$port >/dev/null 2>&1; then
            echo "❌ Failed to free port $port for $service"
            exit 1
        fi
        echo "✅ Port $port freed for $service"
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local port=$1
    local service=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service to be ready on port $port..."
    while ! nc -z localhost $port >/dev/null 2>&1; do
        if [ $attempt -ge $max_attempts ]; then
            echo "❌ $service failed to start on port $port after ${max_attempts}s"
            exit 1
        fi
        sleep 1
        ((attempt++))
    done
    echo "✅ $service is ready on port $port"
}

echo "🚀 Starting RAG Application with Local Models..."
echo "=============================================="

# Check if required commands are available
for cmd in ollama inngest uv nc lsof; do
    if ! command -v $cmd >/dev/null 2>&1; then
        echo "❌ Required command '$cmd' not found. Please install it first."
        exit 1
    fi
done

# Check if Ollama models are available
echo "📋 Checking Ollama models..."
if ! ollama list | grep -q "mxbai-embed-large"; then
    echo "❌ mxbai-embed-large model not found"
    echo "   Run: ollama pull mxbai-embed-large"
    exit 1
fi

if ! ollama list | grep -q "llama3.2"; then
    echo "❌ llama3.2 model not found"
    echo "   Run: ollama pull llama3.2"
    exit 1
fi
echo "✅ Ollama models ready"

# Clean up any existing processes
echo "🧹 Cleaning up any existing services..."
for port in $INNGEST_PORT $QDRANT_PORT $FASTAPI_PORT $STREAMLIT_PORT; do
    check_port $port "service"
done

# Start Inngest dev server
echo "⚡ Starting Inngest dev server..."
check_port $INNGEST_PORT "Inngest"
inngest dev --port $INNGEST_PORT > /dev/null 2>&1 &
INNGEST_PID=$!
echo "   Inngest started (PID: $INNGEST_PID)"
wait_for_service $INNGEST_PORT "Inngest"

# Start Qdrant vector database
echo "🗄️  Starting Qdrant vector database..."
check_port $QDRANT_PORT "Qdrant"
if [[ ! -f "./bin/qdrant" ]]; then
    echo "❌ Qdrant binary not found at ./bin/qdrant"
    echo "   Please ensure Qdrant is installed correctly"
    exit 1
fi
./bin/qdrant > /dev/null 2>&1 &
QDRANT_PID=$!
echo "   Qdrant started (PID: $QDRANT_PID)"
wait_for_service $QDRANT_PORT "Qdrant"

# Start FastAPI backend
echo "🔧 Starting FastAPI backend..."
check_port $FASTAPI_PORT "FastAPI"
uv run uvicorn main:app --reload --host 0.0.0.0 --port $FASTAPI_PORT > /dev/null 2>&1 &
FASTAPI_PID=$!
echo "   FastAPI started (PID: $FASTAPI_PID)"
wait_for_service $FASTAPI_PORT "FastAPI"

# Start Streamlit frontend
echo "🌐 Starting Streamlit frontend..."
check_port $STREAMLIT_PORT "Streamlit"
echo "   📱 Opening in browser: http://localhost:$STREAMLIT_PORT"
echo "   🚀 All services running! Press Ctrl+C to stop."
echo ""

# Start Streamlit (this will block until interrupted)
uv run streamlit run streamlit_app.py --server.port $STREAMLIT_PORT --server.headless true