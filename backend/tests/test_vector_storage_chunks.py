import pytest
from unittest.mock import patch, MagicMock
from app.services.vector_storage import get_document_chunks, VectorStorageError

@patch("app.services.vector_storage._client")
def test_get_document_chunks_success(mock_client):
    from app.config import settings

    # Mock collection existence
    mock_collection = MagicMock()
    mock_collection.name = settings.QDRANT_COLLECTION
    mock_collections_response = MagicMock()
    mock_collections_response.collections = [mock_collection]
    mock_client.get_collections.return_value = mock_collections_response

    # Mock scroll results
    # First page
    record1 = MagicMock()
    record1.payload = {"chunk_index": 2, "text": "Chunk 2"}
    record2 = MagicMock()
    record2.payload = {"chunk_index": 0, "text": "Chunk 0"}
    
    # Second page
    record3 = MagicMock()
    record3.payload = {"chunk_index": 1, "text": "Chunk 1"}
    
    # Return 2 records on first call with offset, 1 record on second call with None offset
    mock_client.scroll.side_effect = [
        ([record1, record2], "next_page_token"),
        ([record3], None)
    ]

    result = get_document_chunks("doc_123")

    assert len(result) == 3
    # Check if they are ordered by chunk_index
    assert result[0]["chunk_index"] == 0
    assert result[1]["chunk_index"] == 1
    assert result[2]["chunk_index"] == 2
    
    # Verify scroll was called with correct filter and args
    assert mock_client.scroll.call_count == 2
    
    first_call_args = mock_client.scroll.call_args_list[0][1]
    assert first_call_args["collection_name"] == settings.QDRANT_COLLECTION
    assert first_call_args["offset"] is None
    assert first_call_args["limit"] == 1000
    assert first_call_args["with_payload"] is True
    assert first_call_args["with_vectors"] is False
    
    # Verify filter matches document_id
    doc_filter = first_call_args["scroll_filter"]
    assert doc_filter.must[0].key == "document_id"
    assert doc_filter.must[0].match.value == "doc_123"

@patch("app.services.vector_storage._client")
def test_get_document_chunks_empty_or_no_collection(mock_client):
    from app.config import settings
    
    # Case 1: Collection doesn't exist
    mock_collections_response = MagicMock()
    mock_collections_response.collections = []
    mock_client.get_collections.return_value = mock_collections_response
    
    result = get_document_chunks("doc_123")
    assert result == []
    assert not mock_client.scroll.called

    # Case 2: Collection exists, but no chunks found
    mock_collection = MagicMock()
    mock_collection.name = settings.QDRANT_COLLECTION
    mock_collections_response.collections = [mock_collection]
    mock_client.get_collections.return_value = mock_collections_response
    
    mock_client.scroll.return_value = ([], None)
    
    result2 = get_document_chunks("doc_123")
    assert result2 == []
    assert mock_client.scroll.called

@patch("app.services.vector_storage._client")
def test_get_document_chunks_error(mock_client):
    from app.config import settings
    
    mock_collection = MagicMock()
    mock_collection.name = settings.QDRANT_COLLECTION
    mock_collections_response = MagicMock()
    mock_collections_response.collections = [mock_collection]
    mock_client.get_collections.return_value = mock_collections_response

    mock_client.scroll.side_effect = Exception("Qdrant connection lost")
    
    with pytest.raises(VectorStorageError, match="Failed to get document chunks: Qdrant connection lost"):
        get_document_chunks("doc_123")
