# RAGLab

RAGLab is an offline Retrieval-Augmented Generation (RAG) experimentation workbench. It is designed to be a local-first application for exploring, building, and evaluating RAG pipelines.

## Main Goal
To provide a clean, modular workbench where users can create knowledge bases, configure document chunking, generate embeddings, and interact with a local LLM—all while having full visibility into the retrieval process and citations.

## Planned Architecture
The system consists of three main components:
1. **Frontend**: React + TypeScript + Vite, styled with Tailwind CSS.
2. **Backend**: Python + FastAPI for REST API and document processing services.
3. **Data/AI Layer**: Local Qdrant for vector storage and local Ollama for embeddings and LLM generation.

## Technology Stack
- **Frontend**: React, TypeScript, Vite, Tailwind CSS, Lucide icons
- **Backend**: Python, FastAPI, Pydantic
- **Document Processing**: PyMuPDF, python-docx
- **Embeddings/LLM**: Ollama (local)
- **Vector Database**: Qdrant (local)

## Current Development Phase
**Phase 3**: Local Embeddings and Semantic Retrieval (Current)
- Integrates local embeddings using Ollama (model: `nomic-embed-text`).
- Integrates local persistent vector storage using Qdrant.
- Implements semantic similarity search via Cosine Similarity.
- The pipeline supports: File Upload -> Parse -> Chunk -> Embed -> Store -> Retrieve.
- *Note: LLM generation is intentionally not implemented yet.*

**Phase 2**: Document Ingestion and Chunking (Completed)
- Support for parsing PDF, TXT, MD, and DOCX files.
- Configurable character-based chunking.
  - Default chunk size: 1000 characters
  - Default chunk overlap: 150 characters
- Metadata preservation (including page numbers for PDFs).
- `POST /api/documents/ingest` endpoint implemented for document upload and processing.

**Phase 1**: Project Initialization (Completed)
- Basic frontend and backend scaffolding
- API health check
- Configuration management
- Minimal frontend connection to backend

## Architecture Overview (Phase 3)
### What are Embeddings?
Embeddings are numerical vector representations of text. They capture the semantic meaning of the words, allowing us to find text that is conceptually similar, even if they don't share the exact same keywords.

### Why Qdrant and Cosine Similarity?
Qdrant is a highly performant vector database. We use its local persistent mode to store the generated embeddings and their metadata without needing Docker or a heavy cloud setup. We use **Cosine Similarity** to measure the angle between two vectors—if the angle is small (similarity near 1), the texts are highly related.

### Environment Variables
Located in `backend/.env`:
- `OLLAMA_BASE_URL`: URL to your local Ollama instance (default: `http://localhost:11434`)
- `EMBEDDING_MODEL`: The Ollama model to use (default: `nomic-embed-text`)
- `QDRANT_COLLECTION`: Name of the vector collection (default: `raglab_documents`)



## Developer Setup

### Prerequisites
- Node.js (v18+)
- Python 3.10+

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Backend Setup
```bash
cd backend
python -m venv venv
# On Windows: venv\Scripts\activate
# On Unix: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Ollama Setup
1. Ensure Ollama is running on your machine (`http://localhost:11434`).
2. Verify the embedding model is installed:
   ```bash
   ollama run nomic-embed-text "test"
   ```

### Running Tests
```bash
cd backend
pytest tests/
```

## API Examples

**Example Ingestion Request:**
```bash
curl -X POST "http://localhost:8000/api/documents/ingest" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample.txt"
```

**Example Retrieval Request:**
```bash
curl -X POST "http://localhost:8000/api/retrieval/search" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the Laplacian matrix?", "top_k": 5}'
```
