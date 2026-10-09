
import os
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings


class VectorStorageError(Exception):
    pass


_client: Optional[QdrantClient] = None
_use_memory: bool = False


def configure_test_mode(use_memory: bool = True) -> None:
    """Configure vector storage to use an in-memory client for tests.

    This must be called before the first client access.
    """
    global _use_memory

    if _client is not None:
        raise VectorStorageError(
            "Cannot change vector storage mode after client initialization."
        )

    _use_memory = use_memory


def get_client() -> QdrantClient:
    """Return the configured Qdrant client, initializing it lazily."""
    global _client

    if _client is not None:
        return _client

    if _use_memory:
        try:
            _client = QdrantClient(location=":memory:")
            return _client
        except Exception as e:
            raise VectorStorageError(
                "Failed to initialize isolated in-memory Qdrant "
                f"for testing: {e}"
            ) from e

    qdrant_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "qdrant_data",
    )
    os.makedirs(qdrant_path, exist_ok=True)

    try:
        _client = QdrantClient(path=qdrant_path)
        return _client
    except Exception as e:
        raise VectorStorageError(
            f"Failed to initialize Qdrant local storage: {e}"
        ) from e


def init_collection_if_needed(dimension: int) -> None:
    """Ensure the collection exists and its vector dimension matches."""
    collection_name = settings.QDRANT_COLLECTION

    try:
        collections = get_client().get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if exists:
            collection_info = get_client().get_collection(collection_name)
            existing_dim = collection_info.config.params.vectors.size

            if existing_dim != dimension:
                raise VectorStorageError(
                    f"Dimension mismatch for collection '{collection_name}'. "
                    f"Expected {existing_dim}, but got {dimension}."
                )
        else:
            get_client().create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
    except VectorStorageError:
        raise
    except Exception as e:
        raise VectorStorageError(
            f"Failed to initialize collection: {e}"
        ) from e


def upsert_points(points: List[PointStruct]) -> None:
    """Insert or update points in the configured Qdrant collection."""
    try:
        get_client().upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )
    except Exception as e:
        raise VectorStorageError(
            f"Failed to upsert points to Qdrant: {e}"
        ) from e


def search_vectors(
    query_vector: List[float],
    top_k: int = 5,
    document_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search Qdrant for similar vectors."""
    try:
        query_filter = None

        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        return get_client().search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
    except Exception as e:
        raise VectorStorageError(
            f"Failed to search vectors: {e}"
        ) from e


def get_all_documents() -> List[Dict[str, Any]]:
    """Get unique documents from Qdrant and count their chunks."""
    try:
        collections = get_client().get_collections().collections

        if not any(
            c.name == settings.QDRANT_COLLECTION for c in collections
        ):
            return []

        documents: Dict[str, Dict[str, Any]] = {}
        offset = None

        while True:
            records, next_page_offset = get_client().scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=["document_id", "filename", "file_type"],
                with_vectors=False,
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
                        "chunk_count": 0,
                    }

                documents[doc_id]["chunk_count"] += 1

            if next_page_offset is None:
                break

            offset = next_page_offset

        return list(documents.values())
    except Exception as e:
        raise VectorStorageError(
            f"Failed to get all documents: {e}"
        ) from e


def get_document_info(document_id: str) -> Optional[Dict[str, Any]]:
    """Return document metadata and chunk count, if the document exists."""
    try:
        collections = get_client().get_collections().collections

        if not any(
            c.name == settings.QDRANT_COLLECTION for c in collections
        ):
            return None

        count_result = get_client().count(
            collection_name=settings.QDRANT_COLLECTION,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )

        if count_result.count == 0:
            return None

        records, _ = get_client().scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            limit=1,
            with_payload=["filename", "file_type"],
            with_vectors=False,
        )

        if not records:
            return None

        payload = records[0].payload or {}

        return {
            "document_id": document_id,
            "filename": payload.get("filename", "Unknown"),
            "file_type": payload.get("file_type", "Unknown"),
            "chunk_count": count_result.count,
        }
    except Exception as e:
        raise VectorStorageError(
            f"Failed to get document info: {e}"
        ) from e


def delete_document(document_id: str) -> None:
    """Delete all chunks belonging to a document."""
    try:
        collections = get_client().get_collections().collections

        if not any(
            c.name == settings.QDRANT_COLLECTION for c in collections
        ):
            return

        get_client().delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
    except Exception as e:
        raise VectorStorageError(
            f"Failed to delete document: {e}"
        ) from e


def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """Retrieve all chunks for a document, ordered by chunk_index."""
    try:
        collections = get_client().get_collections().collections

        if not any(
            c.name == settings.QDRANT_COLLECTION for c in collections
        ):
            return []

        chunks: List[Dict[str, Any]] = []
        offset = None

        while True:
            records, next_page_offset = get_client().scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                if record.payload:
                    chunks.append(record.payload)

            if next_page_offset is None:
                break

            offset = next_page_offset

        chunks.sort(key=lambda chunk: chunk.get("chunk_index", 0))
        return chunks
    except Exception as e:
        raise VectorStorageError(
            f"Failed to get document chunks: {e}"
        ) from e
