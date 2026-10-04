import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.retrieval import search_chunks
from app.schemas import RetrievalRequest

# Mock the embedding function so we can simulate finding chunks based on text.
# The query will be checked inside the mocked search_vectors.

@pytest.mark.asyncio
@patch("app.services.retrieval.embed_text", new_callable=AsyncMock)
@patch("app.services.retrieval.search_vectors")
async def test_multi_document_retrieval_across_collection(mock_search_vectors, mock_embed):
    # Setup mock documents
    doc_a_chunk = {
        "text": "Algebraic connectivity is the second smallest eigenvalue of the Laplacian.",
        "payload": {
            "text": "Algebraic connectivity is the second smallest eigenvalue of the Laplacian.",
            "document_id": "docA",
            "filename": "math.pdf",
            "file_type": "pdf",
            "page": 1,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 74
        }
    }
    
    doc_b_chunk = {
        "text": "BFS explores graph nodes layer by layer using a queue.",
        "payload": {
            "text": "BFS explores graph nodes layer by layer using a queue.",
            "document_id": "docB",
            "filename": "algo.txt",
            "file_type": "txt",
            "page": None,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 54
        }
    }

    mock_embed.return_value = [0.1] * 768
    
    class MockHit:
        def __init__(self, score, payload):
            self.score = score
            self.payload = payload

    def fake_search_vectors(query_vector, top_k, document_id=None):
        hits = []
        if document_id == "docA" or document_id is None:
            hits.append(MockHit(score=0.9, payload=doc_a_chunk["payload"]))
        if document_id == "docB" or document_id is None:
            hits.append(MockHit(score=0.8, payload=doc_b_chunk["payload"]))
        return hits[:top_k]
        
    mock_search_vectors.side_effect = fake_search_vectors

    # 1. Searching across all returns both
    results = await search_chunks("eigenvalue BFS queue", top_k=2)
    assert len(results) == 2
    assert results[0]["metadata"]["document_id"] == "docA"
    assert results[1]["metadata"]["document_id"] == "docB"
    assert results[0]["metadata"]["filename"] == "math.pdf"
    assert results[1]["metadata"]["filename"] == "algo.txt"

    # 2. Searching with document_id filter limits results
    results_a = await search_chunks("eigenvalue BFS queue", top_k=2, document_id="docA")
    assert len(results_a) == 1
    assert results_a[0]["metadata"]["document_id"] == "docA"

    results_b = await search_chunks("eigenvalue BFS queue", top_k=2, document_id="docB")
    assert len(results_b) == 1
    assert results_b[0]["metadata"]["document_id"] == "docB"
