import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)

@patch("app.api.documents.get_document_info")
@patch("app.api.documents.get_document_chunks")
def test_list_document_chunks_success(mock_get_chunks, mock_get_info):
    mock_get_info.return_value = {
        "document_id": "doc123",
        "filename": "test.txt",
        "file_type": "txt",
        "chunk_count": 2
    }
    
    mock_get_chunks.return_value = [
        {
            "text": "Chunk zero",
            "document_id": "doc123",
            "filename": "test.txt",
            "file_type": "txt",
            "page": None,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 10
        },
        {
            "text": "Chunk one",
            "document_id": "doc123",
            "filename": "test.txt",
            "file_type": "txt",
            "page": None,
            "chunk_index": 1,
            "char_start": 10,
            "char_end": 19
        }
    ]
    
    response = client.get("/api/documents/doc123/chunks")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) == 2
    
    assert data[0]["text"] == "Chunk zero"
    assert data[0]["metadata"]["chunk_index"] == 0
    assert data[0]["metadata"]["document_id"] == "doc123"
    
    assert data[1]["text"] == "Chunk one"
    assert data[1]["metadata"]["chunk_index"] == 1
    
    mock_get_info.assert_called_once_with("doc123")
    mock_get_chunks.assert_called_once_with("doc123")

@patch("app.api.documents.get_document_info")
@patch("app.api.documents.get_document_chunks")
def test_list_document_chunks_not_found(mock_get_chunks, mock_get_info):
    mock_get_info.return_value = None
    
    response = client.get("/api/documents/nonexistent/chunks")
    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]
    
    mock_get_info.assert_called_once_with("nonexistent")
    assert not mock_get_chunks.called

@patch("app.api.documents.get_document_info")
@patch("app.api.documents.get_document_chunks")
def test_list_document_chunks_error(mock_get_chunks, mock_get_info):
    from app.services.vector_storage import VectorStorageError
    
    mock_get_info.return_value = {
        "document_id": "doc123",
        "filename": "test.txt",
        "file_type": "txt",
        "chunk_count": 2
    }
    
    mock_get_chunks.side_effect = VectorStorageError("Qdrant connection lost")
    
    response = client.get("/api/documents/doc123/chunks")
    assert response.status_code == 500
    assert "Qdrant connection lost" in response.json()["detail"]
