from typing import List, Dict, Any
from app.services.embeddings import embed_text, EmbeddingError
from app.services.vector_storage import search_vectors, VectorStorageError

class RetrievalError(Exception):
    pass

async def search_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Embeds the query and searches Qdrant for top_k similar chunks.
    """
    if not query or not query.strip():
        raise RetrievalError("Query cannot be empty")
        
    if top_k <= 0:
        raise RetrievalError("top_k must be a positive integer")
        
    # Embed query
    try:
        query_vector = await embed_text(query)
    except EmbeddingError as e:
        raise RetrievalError(f"Embedding failed: {str(e)}")
        
    # Search vector store
    try:
        raw_results = search_vectors(query_vector=query_vector, top_k=top_k)
    except VectorStorageError as e:
        raise RetrievalError(f"Vector search failed: {str(e)}")
        
    # Format results
    formatted_results = []
    for hit in raw_results:
        payload = hit.payload or {}
        # Ensure 'text' is extracted from payload, and the rest is metadata
        text = payload.get("text", "")
        
        # Build metadata excluding 'text'
        metadata = {k: v for k, v in payload.items() if k != "text"}
        
        formatted_results.append({
            "text": text,
            "score": hit.score,
            "metadata": metadata
        })
        
    return formatted_results
