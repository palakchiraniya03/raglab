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

@pytest.mark.asyncio
@patch("app.services.retrieval.embed_text", new_callable=AsyncMock)
@patch("app.services.retrieval.search_vectors")
async def test_search_chunks_threshold(mock_search, mock_embed):
    from app.services.retrieval import search_chunks
    from app.config import settings
    
    mock_embed.return_value = [0.1, 0.2]
    
    # Mock Qdrant results (Testing 0.50 threshold)
    mock_search.return_value = [
        ScoredPoint(id=1, version=1, score=0.60, payload={"text": "High score", "doc_id": "1"}),
        ScoredPoint(id=2, version=1, score=0.50, payload={"text": "Exact threshold", "doc_id": "2"}),
        ScoredPoint(id=3, version=1, score=0.40, payload={"text": "Low score", "doc_id": "3"}),
    ]
    
    results = await search_chunks("query", top_k=3)
    
    # Only 2 should remain
    assert len(results) == 2
    assert results[0]["text"] == "High score"
    assert results[1]["text"] == "Exact threshold"

@pytest.mark.asyncio
@patch("app.services.retrieval.embed_text", new_callable=AsyncMock)
@patch("app.services.retrieval.search_vectors")
async def test_search_chunks_all_below_threshold(mock_search, mock_embed):
    from app.services.retrieval import search_chunks
    from app.config import settings
    
    mock_embed.return_value = [0.1, 0.2]
    
    mock_search.return_value = [
        ScoredPoint(id=3, version=1, score=0.40, payload={"text": "Low score", "doc_id": "3"}),
        ScoredPoint(id=4, version=1, score=0.30, payload={"text": "Lower score", "doc_id": "4"}),
    ]
    
    results = await search_chunks("query", top_k=2)
    assert len(results) == 0

@pytest.mark.asyncio
@patch("app.services.retrieval.embed_text", new_callable=AsyncMock)
@patch("app.services.retrieval.search_vectors")
async def test_search_chunks_retrieval_error(mock_search, mock_embed):
    from app.services.retrieval import search_chunks, VectorStorageError, RetrievalError
    
    mock_embed.return_value = [0.1, 0.2]
    mock_search.side_effect = VectorStorageError("DB failed")
    
    with pytest.raises(RetrievalError, match="Vector search failed"):
        await search_chunks("query")
