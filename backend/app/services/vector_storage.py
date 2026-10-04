import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any

from app.config import settings

class VectorStorageError(Exception):
    pass

# Initialize Qdrant Client (local persistent storage)
# Qdrant will store data in backend/qdrant_data
QDRANT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "qdrant_data")
os.makedirs(QDRANT_PATH, exist_ok=True)

try:
    client = QdrantClient(path=QDRANT_PATH)
except Exception as e:
    raise VectorStorageError(f"Failed to initialize Qdrant local storage: {str(e)}")

def init_collection_if_needed(dimension: int) -> None:
    """
    Ensures the Qdrant collection exists and has the correct dimension.
    """
    collection_name = settings.QDRANT_COLLECTION
    try:
        # Check if collection exists
        collections_response = client.get_collections()
        exists = any(c.name == collection_name for c in collections_response.collections)
        
        if exists:
            # Check dimension matches
            collection_info = client.get_collection(collection_name)
            existing_dim = collection_info.config.params.vectors.size
            if existing_dim != dimension:
                raise VectorStorageError(
                    f"Dimension mismatch for collection '{collection_name}'. "
                    f"Expected {existing_dim}, but got {dimension}."
                )
        else:
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
    except Exception as e:
        if isinstance(e, VectorStorageError):
            raise
        raise VectorStorageError(f"Failed to initialize collection: {str(e)}")

def upsert_points(points: List[PointStruct]) -> None:
    """
    Upsert points into Qdrant collection.
    """
    try:
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points
        )
    except Exception as e:
        raise VectorStorageError(f"Failed to upsert points to Qdrant: {str(e)}")

def search_vectors(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search Qdrant for similar vectors.
    """
    try:
        results = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k
        )
        return results
    except Exception as e:
        raise VectorStorageError(f"Failed to search vectors: {str(e)}")
