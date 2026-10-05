import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional

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

def search_vectors(query_vector: List[float], top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search Qdrant for similar vectors.
    """
    try:
        query_filter = None
        if document_id:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
            
        results = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        return results
    except Exception as e:
        raise VectorStorageError(f"Failed to search vectors: {str(e)}")

from qdrant_client.http.models import Filter, FieldCondition, MatchValue

def get_all_documents() -> List[Dict[str, Any]]:
    """Get unique documents from Qdrant via scroll."""
    try:
        collections = client.get_collections().collections
        if not any(c.name == settings.QDRANT_COLLECTION for c in collections):
            return []

        documents = {}
        offset = None
        while True:
            records, next_page_offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=["document_id", "filename", "file_type"],
                with_vectors=False
            )
            for record in records:
                payload = record.payload or {}
                doc_id = payload.get("document_id")
                if not doc_id:
                    continue
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": payload.get("filename", "Unknown"),
                        "file_type": payload.get("file_type", "Unknown"),
                        "chunk_count": 0
                    }
                documents[doc_id]["chunk_count"] += 1

            if next_page_offset is None:
                break
            offset = next_page_offset

        return list(documents.values())
    except Exception as e:
        raise VectorStorageError(f"Failed to get all documents: {str(e)}")

def get_document_info(document_id: str) -> Optional[Dict[str, Any]]:
    """Check if a document exists and return its metadata and chunk count."""
    try:
        collections = client.get_collections().collections
        if not any(c.name == settings.QDRANT_COLLECTION for c in collections):
            return None

        count_result = client.count(
            collection_name=settings.QDRANT_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            exact=True
        )
        if count_result.count == 0:
            return None

        records, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            limit=1,
            with_payload=["filename", "file_type"],
            with_vectors=False
        )
        if not records:
            return None

        payload = records[0].payload or {}
        return {
            "document_id": document_id,
            "filename": payload.get("filename", "Unknown"),
            "file_type": payload.get("file_type", "Unknown"),
            "chunk_count": count_result.count
        }
    except Exception as e:
        raise VectorStorageError(f"Failed to get document info: {str(e)}")

def delete_document(document_id: str) -> None:
    """Delete all chunks belonging to a document from Qdrant."""
    try:
        collections = client.get_collections().collections
        if not any(c.name == settings.QDRANT_COLLECTION for c in collections):
            return

        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
        )
    except Exception as e:
        raise VectorStorageError(f"Failed to delete document: {str(e)}")

def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """Retrieve all chunks for a specific document, ordered by chunk_index."""
    try:
        collections = client.get_collections().collections
        if not any(c.name == settings.QDRANT_COLLECTION for c in collections):
            return []

        chunks = []
        offset = None
        while True:
            records, next_page_offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                ),
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for record in records:
                if record.payload:
                    chunks.append(record.payload)

            if next_page_offset is None:
                break
            offset = next_page_offset

        # Sort chunks by chunk_index to ensure deterministic ordering
        chunks.sort(key=lambda x: x.get("chunk_index", 0))
        return chunks
    except Exception as e:
        raise VectorStorageError(f"Failed to get document chunks: {str(e)}")
