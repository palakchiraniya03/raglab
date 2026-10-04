import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

from app.main import app
from app.services.generation import generate_text, GenerationError

client = TestClient(app)

@pytest.mark.asyncio
@patch("app.services.generation.httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_generate_text_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"model": "gemma3:1b", "response": "Generated text response"}
    mock_post.return_value = mock_response

    result = await generate_text("Tell me a joke")
    assert result == "Generated text response"
    mock_post.assert_called_once()
    
    call_kwargs = mock_post.call_args.kwargs
    assert "json" in call_kwargs
    assert call_kwargs["json"]["prompt"] == "Tell me a joke"
    assert "model" in call_kwargs["json"]

@pytest.mark.asyncio
async def test_generate_empty_prompt():
    with pytest.raises(GenerationError, match="Cannot generate from empty prompt"):
        await generate_text("   ")

@pytest.mark.asyncio
@patch("app.services.generation.httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_generate_text_model_not_found(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_post.return_value = mock_response

    with pytest.raises(GenerationError, match="not found"):
        await generate_text("test")

@patch("app.api.generation.generate_text", new_callable=AsyncMock)
def test_generate_api_success(mock_generate):
    mock_generate.return_value = "Mocked API generation"
    
    response = client.post(
        "/api/generation/generate",
        json={"prompt": "Test prompt"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Mocked API generation"
    mock_generate.assert_called_once_with("Test prompt")

def test_generate_api_empty_prompt():
    response = client.post(
        "/api/generation/generate",
        json={"prompt": ""}
    )
    assert response.status_code == 400
