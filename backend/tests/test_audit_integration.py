import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json
import os
import tempfile
import hashlib
from unittest.mock import patch, AsyncMock, MagicMock

# Mock QdrantClient before importing app.main to avoid local lock error
patcher = patch("qdrant_client.QdrantClient", autospec=True)
patcher.start()

from app.main import app
from app.services.activity_log import verify_activity_log

client = TestClient(app)

@pytest.fixture
def temp_audit_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "activity.jsonl")
        
        # Patch the LOG_FILE_PATH in activity_log
        with patch('app.services.activity_log.LOG_FILE_PATH', path):
            yield path

@patch("app.api.documents.get_document_info")
@patch("app.api.documents.embed_text", new_callable=AsyncMock)
@patch("app.api.documents.init_collection_if_needed")
@patch("app.api.documents.upsert_points")
def test_successful_new_ingestion_creates_index(mock_upsert, mock_init, mock_embed, mock_get_doc, temp_audit_log):
    mock_get_doc.return_value = None
    mock_embed.return_value = [0.1] * 768
    
    content = b"Integration test indexing."
    response = client.post(
        "/api/documents/ingest",
        files={"file": ("test.txt", content, "text/plain")},
        data={"chunk_size": 1000, "chunk_overlap": 150}
    )
    
    assert response.status_code == 200
    assert os.path.exists(temp_audit_log)
    
    with open(temp_audit_log, 'r', encoding='utf-8') as f:
        events = [json.loads(line) for line in f if line.strip()]
    
    assert len(events) == 1
    assert events[0]["event_type"] == "INDEX"
    assert events[0]["details"]["filename"] == "test.txt"

@patch("app.api.documents.get_document_info")
def test_duplicate_ingestion_creates_duplicate(mock_get_doc, temp_audit_log):
    mock_get_doc.return_value = {
        "document_id": "fake_hash",
        "filename": "test.txt",
        "file_type": "txt",
        "chunk_count": 5
    }
    
    content = b"Duplicate content."
    response = client.post(
        "/api/documents/ingest",
        files={"file": ("test.txt", content, "text/plain")}
    )
    
    assert response.status_code == 200
    assert response.json().get("already_indexed") is True
    
    with open(temp_audit_log, 'r', encoding='utf-8') as f:
        events = [json.loads(line) for line in f if line.strip()]
        
    assert len(events) == 1
    assert events[0]["event_type"] == "DUPLICATE"

@patch("app.api.documents.get_document_info")
@patch("app.api.documents.delete_document")
def test_successful_deletion_creates_delete(mock_delete, mock_get_doc, temp_audit_log):
    mock_get_doc.return_value = {
        "document_id": "doc123",
        "filename": "test.txt",
        "file_type": "txt",
        "chunk_count": 5
    }
    
    response = client.delete("/api/documents/doc123")
    assert response.status_code == 200
    
    with open(temp_audit_log, 'r', encoding='utf-8') as f:
        events = [json.loads(line) for line in f if line.strip()]
        
    assert len(events) == 1
    assert events[0]["event_type"] == "DELETE"

@patch("app.api.documents.get_document_info")
@patch("app.api.documents.delete_document")
def test_deleting_nonexistent_does_not_create_delete(mock_delete, mock_get_doc, temp_audit_log):
    mock_get_doc.return_value = None
    
    response = client.delete("/api/documents/non_existent")
    assert response.status_code == 404
    
    # Check that file either doesn't exist, or is empty
    if os.path.exists(temp_audit_log):
        with open(temp_audit_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 0
    else:
        assert True

@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
@patch("app.api.rag.generate_text", new_callable=AsyncMock)
def test_successful_rag_request_creates_query(mock_generate, mock_search, temp_audit_log):
    mock_search.return_value = [
        {"text": "Sample", "score": 0.9, "metadata": {"document_id": "d1", "filename": "f1"}}
    ]
    mock_generate.return_value = "This is an answer."
    
    response = client.post("/api/rag/ask", json={"query": "hello", "top_k": 3})
    assert response.status_code == 200
    
    with open(temp_audit_log, 'r', encoding='utf-8') as f:
        events = [json.loads(line) for line in f if line.strip()]
        
    assert len(events) == 1
    assert events[0]["event_type"] == "QUERY"
    assert events[0]["details"]["query"] == "hello"
    assert events[0]["details"]["selected_source_count"] == 1

@patch("app.api.documents.get_document_info")
@patch("app.api.documents.embed_text", new_callable=AsyncMock)
def test_failed_ingestion_does_not_create_index(mock_embed, mock_get_doc, temp_audit_log):
    mock_get_doc.return_value = None
    from app.services.embeddings import EmbeddingError
    mock_embed.side_effect = EmbeddingError("Embedding failed")
    
    content = b"Fail me."
    response = client.post(
        "/api/documents/ingest",
        files={"file": ("test.txt", content, "text/plain")},
        data={"chunk_size": 1000, "chunk_overlap": 150}
    )
    
    assert response.status_code != 200
    
    if os.path.exists(temp_audit_log):
        with open(temp_audit_log, 'r', encoding='utf-8') as f:
            events = [line for line in f if line.strip()]
            assert len(events) == 0

@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_failed_rag_does_not_create_query(mock_search, temp_audit_log):
    mock_search.side_effect = Exception("Search error")
    
    response = client.post("/api/rag/ask", json={"query": "fail", "top_k": 3})
    assert response.status_code != 200
    
    if os.path.exists(temp_audit_log):
        with open(temp_audit_log, 'r', encoding='utf-8') as f:
            events = [line for line in f if line.strip()]
            assert len(events) == 0

@patch("app.api.documents.get_document_info")
@patch("app.api.documents.embed_text", new_callable=AsyncMock)
@patch("app.api.documents.init_collection_if_needed")
@patch("app.api.documents.upsert_points")
@patch("app.api.documents.delete_document")
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
@patch("app.api.rag.generate_text", new_callable=AsyncMock)
def test_verify_activity_log_passes_after_sequence(
    mock_generate, mock_search, mock_delete, mock_upsert, mock_init, mock_embed, mock_get_doc, temp_audit_log
):
    # 1. Ingest
    mock_get_doc.return_value = None
    mock_embed.return_value = [0.1] * 768
    client.post("/api/documents/ingest", files={"file": ("test.txt", b"Content", "text/plain")})
    
    # 2. Duplicate
    mock_get_doc.return_value = {"document_id": "hash", "filename": "test.txt", "file_type": "txt", "chunk_count": 1}
    client.post("/api/documents/ingest", files={"file": ("test.txt", b"Content", "text/plain")})
    
    # 3. Query
    mock_search.return_value = [{"text": "Sample", "score": 0.9, "metadata": {"document_id": "d1", "filename": "f1"}}]
    mock_generate.return_value = "Ans"
    client.post("/api/rag/ask", json={"query": "Q", "top_k": 3})
    
    # 4. Delete
    client.delete("/api/documents/doc123")
    
    # Verify chain
    res = verify_activity_log(temp_audit_log)
    assert res["valid"] is True
    assert res["event_count"] == 4
