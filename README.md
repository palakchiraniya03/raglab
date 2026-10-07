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
- 📚 Persistent multi-document knowledge base
- 🔁 Duplicate document detection using SHA-256 document IDs
- 🗑️ Document deletion and knowledge-base management
- 🔍 Retrieval Inspector for understanding which chunks were selected
- 📑 Numbered source evidence with relevant source passages
- 🧪 Evaluation dashboard with retrieval, answer, refusal, and latency metrics
- 🛡️ Tamper-evident hash-chained activity audit trail
- 💾 Persistent local chat history
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

## Evidence and Retrieval Inspector

RAGLab exposes retrieved evidence instead of hiding the retrieval process behind the generated answer.

### Source evidence

Each answer can display numbered source entries containing:

- source document
- page number when available
- chunk index
- retrieval score
- relevant passage from the retrieved chunk

Source numbering is generated structurally from the retrieved results in the frontend. The language model is not asked to invent or generate citation numbers.

This provides a direct connection between the generated answer and the document evidence used to produce it.

### Retrieval Inspector

RAGLab also provides a technical retrieval inspector for understanding why particular chunks were selected.

For retrieved chunks, the inspector can show:

- query
- source document
- page
- chunk index
- semantic similarity
- lexical boost
- final score
- relevance threshold
- whether the chunk was selected

This makes it possible to inspect **how retrieval decisions were made before the context was passed to the generator**.
---

## Persistent Knowledge Base

RAGLab uses local persistent Qdrant storage.

This means indexed documents remain available after restarting the backend.

The knowledge base supports multiple documents simultaneously. Retrieval can search across the complete collection or optionally be restricted to a specific document.

Documents are identified using a SHA-256 hash of their contents, allowing duplicate uploads to be detected without unnecessarily re-embedding the same document.

The knowledge base also supports document management operations such as:

- listing indexed documents
- displaying document and chunk information
- detecting duplicate uploads
- deleting individual documents and their associated vectors

This allows the knowledge base to be maintained throughout the document lifecycle rather than treating ingestion as a one-time operation.
---
## Tamper-Evident Audit Trail

RAGLab maintains a local activity log for important knowledge-base and RAG operations.

The audit trail uses a hash-chained JSONL format. Each event stores:

- timestamp
- event type
- document ID when applicable
- event details
- previous event hash
- current event hash

Each event's SHA-256 hash is calculated from its canonicalized event contents, including the hash of the previous event. This creates a verifiable chain between consecutive events.

Currently recorded operations include:

- document indexing
- duplicate document detection
- document deletion
- RAG queries

The audit log can be independently verified to detect modifications to previously recorded events.

Audit logging is deliberately isolated from the main workflow: an audit logging failure does not prevent the underlying document or RAG operation from completing.

---

## Evaluation

RAGLab includes an evaluation dashboard based on a fixed benchmark derived from an actual source document.

The benchmark contains:

- 9 answerable questions
- 1 intentionally unanswerable question

The evaluation checks:

- whether relevant sources were retrieved
- whether expected terms appeared in the generated answer
- whether the system correctly refused an unsupported question
- response latency

The dashboard provides:

- overall retrieval success
- answer-term success
- refusal success
- average latency
- per-question execution results
- generated answer
- expected terms
- missing answer/source terms
- source count
- per-case diagnosis

### Evaluation robustness

Because generation runs locally through Ollama, individual evaluation cases can take longer than ordinary retrieval operations.

The evaluation runner therefore applies a per-question timeout so that a stalled local generation cannot block the entire evaluation indefinitely.

### Observed result

Evaluation experiments showed that relevant document chunks could be retrieved successfully even when the lightweight Gemma 3 1B model did not always reproduce all expected terms in its final answer.

This distinction between **retrieval quality** and **generation quality** is explicitly surfaced in the evaluation diagnostics.

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
### Retrieval Parameter Tuning

RAGLab exposes several retrieval parameters that control how candidate
chunks are selected and passed to the generation model.

| Parameter | Current Value | Purpose |
|---|---:|---|
| `top_k` | 5 | Number of candidate chunks initially retrieved from Qdrant |
| `LEXICAL_BOOST_WEIGHT` | 0.07 | Small bonus for lexical overlap with the query |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.50 | Minimum final score required for a chunk to be retained |
| `RAG_MAX_CONTEXT_CHUNKS` | 3 | Maximum number of selected chunks passed to the generation model |

#### Lexical Boost Experiment

We evaluated different lexical boost weights while keeping the
knowledge base, embedding model, top-k, threshold, and relevance
criteria fixed.

| Lexical Weight | Retrieval Hit Rate | Precision@K | MRR |
|---:|---:|---:|---:|
| 0.00 | 100.0% | 0.741 | 1.000 |
| 0.03 | 100.0% | 0.741 | 1.000 |
| 0.05 | 100.0% | 0.741 | 1.000 |
| 0.07 | 100.0% | 0.741 | 1.000 |
| 0.10 | 100.0% | 0.778 | 1.000 |

The experiment showed that retrieval remained stable across the tested
weights. The current value of 0.07 is retained as a conservative
lexical contribution so that semantic similarity remains the dominant
retrieval signal. The experiment also showed that 0.10 improved
Precision@K on this benchmark, demonstrating that the lexical weight
is a tunable parameter rather than a universally optimal value.

#### Context Size

RAGLab initially retrieves up to 5 candidate chunks from Qdrant,
filters and ranks them, and passes at most 3 selected chunks to the
generation model. This provides enough context for multi-chunk and
multi-document reasoning while limiting unnecessary context for the
lightweight Gemma 3 1B model.

#### Retrieval Threshold

A final-score threshold of 0.50 is used to discard weak retrieval
results before generation. This is an empirically chosen, tunable
parameter rather than a universally optimal value. It also allows the
system to refuse questions when no sufficiently relevant document
evidence is found.

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
