import httpx
from app.config import settings

class GenerationError(Exception):
    pass

async def generate_text(prompt: str) -> str:
    """
    Calls the local Ollama instance to generate text for the given prompt.
    """
    if not prompt or not prompt.strip():
        raise GenerationError("Cannot generate from empty prompt")

    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": settings.GENERATION_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
    except httpx.RequestError as e:
        raise GenerationError(f"Failed to connect to Ollama: {str(e)}")

    if response.status_code == 404:
        raise GenerationError(f"Generation model '{settings.GENERATION_MODEL}' not found. Please install it.")
    elif response.status_code != 200:
        raise GenerationError(f"Ollama returned an error: {response.text}")

    data = response.json()
    if "response" not in data:
        raise GenerationError("Malformed generation response from Ollama")

    return data["response"]
