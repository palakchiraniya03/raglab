import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import app
import hashlib

client = TestClient(app)

def test_chunking_service():
    from app.services.chunking import chunk_text
    
    text = "A" * 2000
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=100)
    
    assert len(chunks) == 3
    assert chunks[0][1] == 0
    assert chunks[0][2] == 1000
    assert chunks[1][1] == 900
    assert chunks[1][2] == 1900
    assert chunks[2][1] == 1800
    assert chunks[2][2] == 2000
    
    for chunk, start, end in chunks:
        assert len(chunk) == end - start
        assert text[start:end] == chunk

@patch("app.api.documents.embed_text", new_callable=AsyncMock)
@patch("app.api.documents.init_collection_if_needed")
@patch("app.api.documents.upsert_points")
def test_ingest_txt(mock_upsert, mock_init, mock_embed):
    # Mock embedding returns a vector of 768 dims
    mock_embed.return_value = [0.1] * 768
    
    content = b"This is a simple test document.\nIt has multiple lines.\n"
    response = client.post(
        "/api/documents/ingest",
        files={"file": ("test.txt", content, "text/plain")},
        data={"chunk_size": 15, "chunk_overlap": 5}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["file_type"] == "txt"
    assert data["total_characters"] == len(content.decode('utf-8'))
    assert len(data["chunks"]) > 0
    assert data["embedded_chunks"] == len(data["chunks"])
    assert data["chunks"][0]["metadata"]["page"] is None
    
    # Check document ID stability
    expected_id = hashlib.sha256(content).hexdigest()
    assert data["document_id"] == expected_id
    
    assert mock_embed.called
    assert mock_init.called
    assert mock_upsert.called


def test_ingest_unsupported_file():
    response = client.post(
        "/api/documents/ingest",
        files={"file": ("test.xyz", b"fake content", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_ingest_empty_document():
    response = client.post(
        "/api/documents/ingest",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert "File is empty" in response.json()["detail"]
