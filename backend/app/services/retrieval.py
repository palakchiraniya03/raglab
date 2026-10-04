import re
from typing import List, Dict, Any
from app.services.embeddings import embed_text, EmbeddingError
from app.services.vector_storage import search_vectors, VectorStorageError

class RetrievalError(Exception):
    pass

STOP_WORDS = {"what", "is", "the", "of", "a", "an", "are", "was", "were",
              "to", "in", "on", "for", "and", "or", "how", "why", "which"}
MAX_LEXICAL_BOOST = 0.07

def _calculate_lexical_boost(query: str, chunk_text: str) -> float:
    """
    Calculates a small lexical relevance boost based on the fraction of
    meaningful query tokens that appear in the chunk text.
    """
    query_clean = re.sub(r'[^\w\s]', '', query.lower())
    chunk_clean = re.sub(r'[^\w\s]', '', chunk_text.lower())

    query_tokens = {w for w in query_clean.split() if len(w) >= 3 and w not in STOP_WORDS}
    if not query_tokens:
        return 0.0

    chunk_tokens = set(chunk_clean.split())
    overlap = len(query_tokens.intersection(chunk_tokens))

    return (overlap / len(query_tokens)) * MAX_LEXICAL_BOOST

from typing import List, Dict, Any, Optional

async def search_chunks(query: str, top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
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
        raw_results = search_vectors(query_vector=query_vector, top_k=top_k, document_id=document_id)
    except VectorStorageError as e:
        raise RetrievalError(f"Vector search failed: {str(e)}")

    # Format results
    from app.config import settings

    formatted_results = []

    for hit in raw_results:
        payload = hit.payload or {}
        text = payload.get("text", "")

        lexical_boost = _calculate_lexical_boost(query, text)
        final_score = hit.score + lexical_boost

        if final_score < settings.RETRIEVAL_SCORE_THRESHOLD:
            continue

        # Build metadata excluding 'text'
        metadata = {k: v for k, v in payload.items() if k != "text"}

        formatted_results.append({
            "text": text,
            # Returning final_score as 'score' so that it correctly passes RAG boundaries
            # downstream which expect score >= 0.50
            "score": final_score,
            "semantic_score": hit.score,
            "lexical_boost": lexical_boost,
            "final_score": final_score,
            "selected": final_score >= settings.RETRIEVAL_SCORE_THRESHOLD,
            "metadata": metadata
        })

    # Re-sort by final_score descending since lexical boosts might alter ordering
    formatted_results.sort(key=lambda x: x["score"], reverse=True)

    return formatted_results
