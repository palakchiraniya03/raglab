import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import app
from qdrant_client.http.models import ScoredPoint

client = TestClient(app)

@patch("app.api.retrieval.search_chunks", new_callable=AsyncMock)
def test_retrieval_api(mock_search_chunks):
    # Mock the return value of search_chunks
    mock_search_chunks.return_value = [
        {
            "text": "The Laplacian matrix is used in graph theory.",
            "score": 0.95,
            "metadata": {
                "document_id": "test_doc_1",
                "filename": "graph_theory.txt",
                "file_type": "txt",
                "page": None,
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 45
            }
        }
    ]
    
    response = client.post(
        "/api/retrieval/search",
        json={"query": "What is the Laplacian matrix?", "top_k": 3}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is the Laplacian matrix?"
    assert len(data["results"]) == 1
    assert data["results"][0]["text"] == "The Laplacian matrix is used in graph theory."
    assert data["results"][0]["score"] == 0.95
    assert data["results"][0]["metadata"]["document_id"] == "test_doc_1"

def test_retrieval_empty_query():
    response = client.post(
        "/api/retrieval/search",
        json={"query": "", "top_k": 5}
    )
    assert response.status_code == 400

def test_retrieval_invalid_top_k():
    response = client.post(
        "/api/retrieval/search",
        json={"query": "valid query", "top_k": 0}
    )
    assert response.status_code == 400
