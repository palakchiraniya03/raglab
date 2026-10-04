import httpx
from typing import List
from fastapi import HTTPException, status
from app.config import settings

class EmbeddingError(Exception):
    pass

async def embed_text(text: str) -> List[float]:
    """
    Calls the local Ollama instance to generate an embedding for the given text.
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty text")
        
    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
    payload = {
        "model": settings.EMBEDDING_MODEL,
        "prompt": text
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
    except httpx.RequestError as e:
        raise EmbeddingError(f"Failed to connect to Ollama: {str(e)}")
        
    if response.status_code == 404:
        raise EmbeddingError(f"Embedding model '{settings.EMBEDDING_MODEL}' not found. Please install it.")
    elif response.status_code != 200:
        raise EmbeddingError(f"Ollama returned an error: {response.text}")
        
    data = response.json()
    if "embedding" not in data:
        raise EmbeddingError("Malformed embedding response from Ollama")
        
    embedding = data["embedding"]
    if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
        raise EmbeddingError("Embedding is not a valid list of floats")
        
    return [float(x) for x in embedding]
