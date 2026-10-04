import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.main import app
from app.services.retrieval import RetrievalError
from app.services.generation import GenerationError

client = TestClient(app)

@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_success(mock_search_chunks, mock_generate_text):
    mock_search_chunks.return_value = [
        {
            "text": "The Laplacian matrix is L = D - A.",
            "score": 0.95,
            "metadata": {
                "document_id": "test_doc_1",
                "filename": "math.txt",
                "file_type": "txt",
                "page": None,
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 45
            }
        },
        {
            "text": "It has properties related to graph connectivity.",
            "score": 0.85,
            "metadata": {
                "document_id": "test_doc_2",
                "filename": "graph.pdf",
                "file_type": "pdf",
                "page": 2,
                "chunk_index": 1,
                "char_start": 100,
                "char_end": 146
            }
        }
    ]
    
    mock_generate_text.return_value = "The Laplacian matrix is L = D - A and it relates to graph connectivity."
    
    response = client.post(
        "/api/rag/ask",
        json={"query": "What is the Laplacian matrix?", "top_k": 2}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["answer"] == "The Laplacian matrix is L = D - A and it relates to graph connectivity."
    assert len(data["sources"]) == 2
    assert data["sources"][0]["text"] == "The Laplacian matrix is L = D - A."
    
    mock_search_chunks.assert_called_once_with(query="What is the Laplacian matrix?", top_k=2)
    mock_generate_text.assert_called_once()
    
    # Verify prompt construction
    prompt = mock_generate_text.call_args[1]["prompt"]
    assert "[Source 1]" in prompt
    assert "math.txt" in prompt
    assert "[Source 2]" in prompt
    assert "graph.pdf" in prompt
    assert "Page: 2" in prompt
    assert "What is the Laplacian matrix?" in prompt


@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_no_context(mock_search_chunks, mock_generate_text):
    mock_search_chunks.return_value = []
    
    response = client.post(
        "/api/rag/ask",
        json={"query": "What is the meaning of life?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "could not find any relevant information" in data["answer"].lower()
    assert len(data["sources"]) == 0
    
    mock_search_chunks.assert_called_once()
    mock_generate_text.assert_not_called()


@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_retrieval_failure(mock_search_chunks):
    mock_search_chunks.side_effect = RetrievalError("Database connection lost")
    
    response = client.post(
        "/api/rag/ask",
        json={"query": "test"}
    )
    
    assert response.status_code == 500
    assert "Retrieval failed" in response.json()["detail"]


@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_generation_failure(mock_search_chunks, mock_generate_text):
    mock_search_chunks.return_value = [
        {
            "text": "Some text",
            "score": 0.9,
            "metadata": {
                "document_id": "1", "filename": "1.txt", "file_type": "txt",
                "page": None, "chunk_index": 0, "char_start": 0, "char_end": 10
            }
        }
    ]
    mock_generate_text.side_effect = GenerationError("Ollama timeout")
    
    response = client.post(
        "/api/rag/ask",
        json={"query": "test"}
    )
    
    assert response.status_code == 502
    assert "Generation failed" in response.json()["detail"]

def test_rag_empty_query():
    response = client.post(
        "/api/rag/ask",
        json={"query": "", "top_k": 5}
    )
    assert response.status_code == 400

def test_rag_invalid_top_k():
    response = client.post(
        "/api/rag/ask",
        json={"query": "valid query", "top_k": 0}
    )
    assert response.status_code == 400

@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_max_context_chunks(mock_search_chunks, mock_generate_text):
    # Mock returning 5 relevant chunks
    mock_search_chunks.return_value = [
        {"text": f"Chunk {i}", "score": 0.9, "metadata": {"document_id": "1", "filename": "1.txt", "file_type": "txt", "page": None, "chunk_index": i, "char_start": 0, "char_end": 10}}
        for i in range(5)
    ]
    
    mock_generate_text.return_value = "Mocked answer"
    
    response = client.post(
        "/api/rag/ask",
        json={"query": "test query", "top_k": 5}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    from app.config import settings
    # The API should limit to RAG_MAX_CONTEXT_CHUNKS (which defaults to 3)
    expected_limit = settings.RAG_MAX_CONTEXT_CHUNKS
    
    assert len(data["sources"]) == expected_limit
    assert data["sources"][0]["text"] == "Chunk 0"
    assert data["sources"][1]["text"] == "Chunk 1"
    assert data["sources"][2]["text"] == "Chunk 2"
    
    prompt = mock_generate_text.call_args[1]["prompt"]
    assert "Chunk 0" in prompt
    assert "Chunk 1" in prompt
    assert "Chunk 2" in prompt
    assert "Chunk 3" not in prompt
    assert "Chunk 4" not in prompt

@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_fewer_than_max_chunks(mock_search_chunks, mock_generate_text):
    # Mock returning 2 chunks (less than max 3)
    mock_search_chunks.return_value = [
        {"text": f"Chunk {i}", "score": 0.9, "metadata": {"document_id": "1", "filename": "1.txt", "file_type": "txt", "page": None, "chunk_index": i, "char_start": 0, "char_end": 10}}
        for i in range(2)
    ]
    
    mock_generate_text.return_value = "Mocked answer"
    
    response = client.post(
        "/api/rag/ask",
        json={"query": "test query", "top_k": 5}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 2
