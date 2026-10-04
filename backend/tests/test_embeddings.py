import pytest
from unittest.mock import patch, AsyncMock
import httpx
from app.services.embeddings import embed_text, EmbeddingError

@pytest.mark.asyncio
@patch("app.services.embeddings.httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_embed_text_success(mock_post):
    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_post.return_value = mock_response

    result = await embed_text("Hello world")
    assert result == [0.1, 0.2, 0.3]
    mock_post.assert_called_once()
    
    # Verify the payload sent to Ollama
    call_kwargs = mock_post.call_args.kwargs
    assert "json" in call_kwargs
    assert call_kwargs["json"]["prompt"] == "Hello world"
    assert "model" in call_kwargs["json"]

@pytest.mark.asyncio
async def test_embed_empty_text():
    with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
        await embed_text("   ")

@pytest.mark.asyncio
@patch("app.services.embeddings.httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_embed_text_model_not_found(mock_post):
    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_post.return_value = mock_response

    with pytest.raises(EmbeddingError, match="not found"):
        await embed_text("test")

@pytest.mark.asyncio
@patch("app.services.embeddings.httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_embed_text_connection_error(mock_post):
    mock_post.side_effect = httpx.RequestError("Connection failed")

    with pytest.raises(EmbeddingError, match="Failed to connect to Ollama"):
        await embed_text("test")
