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
**Phase 2**: Document Ingestion and Chunking (Current)
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
