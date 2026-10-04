# RAGLab

**RAGLab** is an offline Retrieval-Augmented Generation (RAG) workbench for building, inspecting, and experimenting with document-based question answering using local AI models.

The goal is simple:

> **Bring your documents. Build a local knowledge base. Retrieve relevant information. Chat with your documents. Understand why the system answered the way it did.**

RAGLab is designed to run locally without requiring a cloud LLM or hosted vector database.

---

## Features

- 📄 Upload and process **PDF, TXT, Markdown, and DOCX** documents
- ✂️ Character-based document chunking with configurable overlap
- 🧠 Local text embeddings using **Ollama + `nomic-embed-text`**
- 🔎 Semantic retrieval using **Qdrant**
- 🏷️ Metadata-aware retrieval with document, page, and chunk information
- ⚡ Lexical relevance boosting alongside semantic similarity
- 🤖 Local answer generation using **Ollama + Gemma 3 1B**
- 💬 ChatGPT-style document question-answering interface
- 📚 Persistent local knowledge base
- 🔍 Retrieval Inspector for understanding which chunks were selected
- 📑 Source citations and relevant source passages
- 🧪 Lightweight RAG evaluation workflow
- 🔬 Experiments with retrieval `top_k` and document chunk size
- 📴 Fully local/offline AI pipeline after model setup

---

## Architecture

```text
                         RAGLab
                           │
              ┌────────────┴────────────┐
              │                         │
        React Frontend             FastAPI Backend
              │                         │
              │              ┌──────────┴──────────┐
              │              │                     │
              │        Document Pipeline      RAG Pipeline
              │              │                     │
              │       Parse → Chunk          Query Embedding
              │              │                     │
              │          Embeddings              │
              │              │                     │
              │          Qdrant ◄────── Retrieval
              │                                    │
              │                              Context Builder
              │                                    │
              │                              Gemma 3 1B
              │                                    │
              └──────────── Answer + Sources ◄─────┘
```

### Document ingestion

```text
PDF / TXT / MD / DOCX
          ↓
      Parse text
          ↓
       Chunk text
          ↓
    Generate embeddings
          ↓
   Store in local Qdrant
```

### Question answering

```text
User question
      ↓
Generate query embedding
      ↓
Semantic retrieval
      ↓
Lexical relevance adjustment
      ↓
Relevant chunks
      ↓
Context builder
      ↓
Gemma 3 1B
      ↓
Answer + source evidence
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Styling | Tailwind CSS |
| Icons | Lucide React |
| Backend | Python, FastAPI |
| Validation | Pydantic |
| PDF Processing | PyMuPDF |
| DOCX Processing | python-docx |
| Embeddings | Ollama + `nomic-embed-text` |
| LLM | Ollama + `gemma3:1b` |
| Vector Database | Qdrant |
| Testing | Pytest |

---

## RAG Pipeline

### 1. Document ingestion

Uploaded documents are parsed according to their file type.

Supported formats:

- PDF
- TXT
- Markdown
- DOCX

Each document receives a deterministic SHA-256 `document_id`.

---

### 2. Chunking

Documents are divided into overlapping character-based chunks.

Current defaults:

```text
Chunk size:     1000 characters
Chunk overlap:   150 characters
```

Each chunk retains metadata such as:

- document ID
- filename
- file type
- page number
- chunk index
- character start/end positions

---

### 3. Embeddings

Each chunk is converted into a vector using:

```text
Ollama
└── nomic-embed-text
```

The embeddings are stored locally in Qdrant.

---

### 4. Retrieval

When the user asks a question:

1. The question is embedded using the same embedding model.
2. Qdrant performs semantic similarity search.
3. Low-scoring results are filtered using a relevance threshold.
4. A small lexical overlap boost helps queries containing important document terms.
5. The most relevant chunks are passed to the RAG context builder.

---

### 5. Generation

The selected document chunks are provided to:

```text
Ollama
└── Gemma 3 1B
```

The model is instructed to answer using the retrieved document context rather than relying on outside knowledge.

If the retrieved context does not contain the answer, RAGLab can return:

```text
The information is not available in the provided documents.
```

---

## Retrieval Inspector

RAGLab exposes the retrieval process instead of hiding it.

For retrieved chunks, the frontend can show information such as:

- source document
- page
- chunk index
- semantic similarity
- lexical boost
- final score
- relevance threshold
- whether the chunk was selected

This makes it possible to inspect **why particular document passages were provided to the generator**.

---

## Persistent Knowledge Base

RAGLab uses local persistent Qdrant storage.

This means indexed documents remain available after restarting the backend.

Documents are identified using a SHA-256 hash of their contents, allowing duplicate uploads to be detected without unnecessarily re-embedding the same document.

---

## Evaluation

RAGLab includes a small evaluation workflow based on questions derived from an actual source document.

The evaluation contains:

- 9 answerable questions
- 1 intentionally unanswerable question

The evaluation checks:

- whether relevant sources were retrieved
- whether expected terms appeared in the generated answer
- whether the system correctly refused an unsupported question
- response latency

### Observed result

The evaluation demonstrated that relevant document chunks were successfully retrieved for all answerable questions.

The main limitation observed was **local generation quality from the lightweight Gemma 3 1B model**, rather than document retrieval.

This distinction was verified by checking whether the expected information was present in the retrieved source chunks.

---

## RAG Experiments

### Experiment 1 — Retrieval `top_k`

The same query was tested with:

```text
top_k = 1
top_k = 3
top_k = 5
```

Increasing `top_k` retrieved more context but also introduced less relevant chunks.

For the tested query, the first result contained the exact definition while later results increasingly contained unrelated material.

**Conclusion:** RAGLab retains a small context size rather than blindly passing many retrieved chunks to the generator.

---

### Experiment 2 — Chunk Size

The document chunking strategy was examined using different chunk sizes:

```text
500 characters
1000 characters
1500 characters
```

The existing configuration:

```text
1000 character chunks
150 character overlap
```

was retained because it preserved relevant passages effectively for the tested document.

No unnecessary change was made to the production configuration based on this experiment.

---

## Project Structure

```text
RAGLab/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── documents.py
│   │   │   ├── retrieval.py
│   │   │   ├── generation.py
│   │   │   └── rag.py
│   │   │
│   │   ├── services/
│   │   │   ├── parsing.py
│   │   │   ├── chunking.py
│   │   │   ├── embeddings.py
│   │   │   ├── retrieval.py
│   │   │   ├── vector_storage.py
│   │   │   └── generation.py
│   │   │
│   │   ├── config.py
│   │   ├── schemas.py
│   │   └── main.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── questions.json
│   │
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   └── src/
│
├── docs/
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama

---

### 1. Clone the repository

```bash
git clone https://github.com/palakchiraniya03/raglab.git
cd RAGLab
```

---

### 2. Set up the backend

```bash
cd backend
python -m venv venv
```

#### Windows

```powershell
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 3. Install local Ollama models

Make sure Ollama is running, then install:

```bash
ollama pull nomic-embed-text
ollama pull gemma3:1b
```

Verify:

```bash
ollama list
```

---

### 4. Start the backend

From the `backend` directory:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

### 5. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open the local Vite URL shown in the terminal.

---

## Running Tests

From the project root:

```powershell
.\backend\venv\Scripts\python.exe -m pytest backend/tests -q
```

The project currently includes unit and API-level tests covering document ingestion, retrieval, generation, and RAG behavior.

---

## API Endpoints

### Health Check

```text
GET /api/health
```

### Document Ingestion

```text
POST /api/documents/ingest
```

Supports:

```text
PDF
TXT
MD
DOCX
```

### List Documents

```text
GET /api/documents
```

### Retrieval

```text
POST /api/retrieval/search
```

Example:

```json
{
  "query": "What is the Laplacian matrix?",
  "top_k": 5
}
```

### Local Generation

```text
POST /api/generation/generate
```

### RAG Question Answering

```text
POST /api/rag/ask
```

---

## Design Principles

RAGLab intentionally follows a few principles:

### Local-first

Documents, embeddings, vector storage, and generation can all run locally.

### Inspectable

The system exposes retrieval information rather than treating RAG as a black box.

### Modular

Parsing, chunking, embeddings, retrieval, generation, and API layers are separated so individual components can be experimented with independently.

### Lightweight

The system is designed to work with relatively lightweight local models and does not require a large cloud infrastructure stack.

### Experiment-driven

Changes to retrieval and generation behavior are tested on real document data rather than being added only for complexity.

---

## Current Status

RAGLab currently supports the complete local RAG workflow:

```text
Document
   ↓
Parsing
   ↓
Chunking
   ↓
Embeddings
   ↓
Qdrant
   ↓
Retrieval
   ↓
Context Selection
   ↓
Gemma 3 1B
   ↓
Answer + Sources
```

The project is now in its **final polish and documentation stage**.