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

    mock_embed.assert_called_once_with("query")

    # Only 2 should remain
    assert len(results) == 2
    assert results[0]["text"] == "High score"
    assert results[0]["semantic_score"] == 0.60
    assert results[0]["final_score"] == results[0]["score"]
    assert results[0]["selected"] is True
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

    mock_embed.assert_called_once_with("query")
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

    mock_embed.assert_called_once_with("query")

@pytest.mark.asyncio
@patch("app.services.retrieval.embed_text", new_callable=AsyncMock)
@patch("app.services.retrieval.search_vectors")
async def test_search_chunks_lexical_fiedler(mock_search, mock_embed):
    from app.services.retrieval import search_chunks

    mock_embed.return_value = [0.1, 0.2]

    # Fiedler scenario: Semantic score 0.45, text contains Fiedler
    mock_search.return_value = [
        ScoredPoint(id=1, version=1, score=0.45, payload={"text": "The Fiedler value is the second smallest eigenvalue.", "doc_id": "1"})
    ]

    results = await search_chunks("What is the Fiedler value?", top_k=3)

    # 0.45 + boost (fiedler and value match. "fiedler", "value" are meaningful tokens)
    # query tokens: fiedler, value. chunk tokens: fiedler, value -> overlap = 2/2 = 1.0 -> boost = 0.07. 0.45 + 0.07 = 0.52 >= 0.50
    assert len(results) == 1
    assert results[0]["score"] == 0.52
    assert results[0]["semantic_score"] == 0.45
    assert results[0]["lexical_boost"] == 0.07
    assert results[0]["final_score"] == 0.52
    assert results[0]["selected"] is True

@pytest.mark.asyncio
@patch("app.services.retrieval.embed_text", new_callable=AsyncMock)
@patch("app.services.retrieval.search_vectors")
async def test_search_chunks_lexical_france(mock_search, mock_embed):
    from app.services.retrieval import search_chunks

    mock_embed.return_value = [0.1, 0.2]

    # France scenario: Semantic score 0.46, unrelated text
    mock_search.return_value = [
        ScoredPoint(id=1, version=1, score=0.46, payload={"text": "This is a math chunk with no relevant terms.", "doc_id": "1"})
    ]

    results = await search_chunks("What is the capital of France?", top_k=3)

    # query tokens: capital, france. Overlap = 0. boost = 0. 0.46 < 0.50 -> dropped
    assert len(results) == 0

@pytest.mark.asyncio
@patch("app.services.retrieval.embed_text", new_callable=AsyncMock)
@patch("app.services.retrieval.search_vectors")
async def test_search_chunks_lexical_stopwords(mock_search, mock_embed):
    from app.services.retrieval import search_chunks

    mock_embed.return_value = [0.1, 0.2]

    # Stopword scenario
    mock_search.return_value = [
        ScoredPoint(id=1, version=1, score=0.49, payload={"text": "Some text containing nothing important.", "doc_id": "1"})
    ]

    results = await search_chunks("What is the?", top_k=3)

    # No meaningful tokens, boost = 0. 0.49 < 0.50 -> dropped
    assert len(results) == 0

def test_calculate_lexical_boost_max():
    from app.services.retrieval import _calculate_lexical_boost

    boost = _calculate_lexical_boost("test query", "test query")
    assert boost == 0.07

    # Partial
    boost2 = _calculate_lexical_boost("test query another", "test query missing")
    assert abs(boost2 - (2/3 * 0.07)) < 1e-6
