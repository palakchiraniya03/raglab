from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
import hashlib
import os
from typing import Optional

from app.schemas import IngestResponse, ChunkResponse, ChunkMetadata, DocumentItem
from app.services.parsing import parse_document, DocumentParsingError
from app.services.chunking import chunk_text
from app.services.embeddings import embed_text, EmbeddingError
from app.services.vector_storage import init_collection_if_needed, upsert_points, VectorStorageError, get_all_documents, get_document_info, delete_document
from app.services.activity_log import log_activity
from qdrant_client.http.models import PointStruct
from app.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

@router.get("", response_model=list[DocumentItem])
@router.get("/", response_model=list[DocumentItem], include_in_schema=False)
async def list_documents():
    try:
        return get_all_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150)
):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        content = await file.read()
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read file")

    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty")

    # Generate stable document_id using SHA-256 hash of file content
    document_id = hashlib.sha256(content).hexdigest()

    # Check if already indexed
    try:
        existing_doc = get_document_info(document_id)
        if existing_doc:
            try:
                log_activity(
                    event_type="DUPLICATE",
                    document_id=existing_doc["document_id"],
                    details={
                        "filename": existing_doc["filename"],
                        "file_type": existing_doc["file_type"],
                        "chunk_count": existing_doc["chunk_count"]
                    }
                )
            except Exception as e:
                print(f"Audit log failed: {e}")
                
            return IngestResponse(
                document_id=existing_doc["document_id"],
                filename=existing_doc["filename"],
                file_type=existing_doc["file_type"],
                total_characters=0,
                total_chunks=existing_doc["chunk_count"],
                embedded_chunks=existing_doc["chunk_count"],
                collection=settings.QDRANT_COLLECTION,
                chunks=[],
                already_indexed=True
            )
    except VectorStorageError:
        pass  # proceed if checking fails

    # Parse document
    try:
        parsed_pages = parse_document(content, ext[1:])  # remove the dot
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DocumentParsingError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    if not parsed_pages:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No text could be extracted from the document")

    # Chunking
    chunks_response = []
    global_chunk_index = 0
    total_characters = 0

    for page in parsed_pages:
        page_text = page.text
        if not page_text.strip():
            continue

        page_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for text, start, end in page_chunks:
            chunks_response.append(
                ChunkResponse(
                    text=text,
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        filename=file.filename,
                        file_type=ext[1:],
                        page=page.page_num,
                        chunk_index=global_chunk_index,
                        char_start=start,
                        char_end=end
                    )
                )
            )
            global_chunk_index += 1

        total_characters += len(page_text)

    if not chunks_response:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No valid text chunks could be generated")

    # Phase 3: Embed and Store in Qdrant
    import uuid

    points_to_upsert = []

    # Try embedding the first chunk to determine dimension
    try:
        first_embedding = await embed_text(chunks_response[0].text)
        dimension = len(first_embedding)
    except EmbeddingError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))

    # Initialize collection based on detected dimension
    try:
        init_collection_if_needed(dimension)
    except VectorStorageError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # Prepare point for first chunk
    payload_dict = chunks_response[0].metadata.model_dump()
    payload_dict["text"] = chunks_response[0].text

    # Deterministic point ID: SHA-256 of document_id + ":" + chunk_index
    # We take the first 32 hex chars to form a valid UUID-like hex string for Qdrant,
    # Qdrant accepts UUID or unsigned integer. We can hash to a UUID string.
    def get_point_id(doc_id: str, c_index: int) -> str:
        hash_str = hashlib.sha256(f"{doc_id}:{c_index}".encode()).hexdigest()
        # Format as UUID: 8-4-4-4-12
        return f"{hash_str[:8]}-{hash_str[8:12]}-{hash_str[12:16]}-{hash_str[16:20]}-{hash_str[20:32]}"

    points_to_upsert.append(
        PointStruct(
            id=get_point_id(document_id, chunks_response[0].metadata.chunk_index),
            vector=first_embedding,
            payload=payload_dict
        )
    )

    # Embed and prepare remaining chunks
    for cr in chunks_response[1:]:
        try:
            vec = await embed_text(cr.text)
        except EmbeddingError as e:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Embedding failed at chunk {cr.metadata.chunk_index}: {str(e)}")

        payload_dict = cr.metadata.model_dump()
        payload_dict["text"] = cr.text

        points_to_upsert.append(
            PointStruct(
                id=get_point_id(document_id, cr.metadata.chunk_index),
                vector=vec,
                payload=payload_dict
            )
        )

    # Upsert all points to Qdrant
    try:
        upsert_points(points_to_upsert)
    except VectorStorageError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))



    try:
        log_activity(
            event_type="INDEX",
            document_id=document_id,
            details={
                "filename": file.filename,
                "file_type": ext[1:],
                "chunk_count": len(chunks_response)
            }
        )
    except Exception as e:
        print(f"Audit log failed: {e}")

    return IngestResponse(
        document_id=document_id,
        filename=file.filename,
        file_type=ext[1:],
        total_characters=total_characters,
        total_chunks=len(chunks_response),
        embedded_chunks=len(points_to_upsert),
        collection=settings.QDRANT_COLLECTION,
        chunks=chunks_response
    )

@router.delete("/{document_id}")
async def delete_indexed_document(document_id: str):
    try:
        existing_doc = get_document_info(document_id)
        if not existing_doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

        delete_document(document_id)
        
        try:
            log_activity(
                event_type="DELETE",
                document_id=document_id,
                details={
                    "filename": existing_doc["filename"],
                    "file_type": existing_doc["file_type"],
                    "deleted_chunk_count": existing_doc["chunk_count"]
                }
            )
        except Exception as e:
            print(f"Audit log failed: {e}")

        return {
            "document_id": document_id,
            "deleted": True,
            "deleted_chunk_count": existing_doc["chunk_count"]
        }
    except VectorStorageError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

from app.services.vector_storage import get_document_chunks

@router.get("/{document_id}/chunks", response_model=list[ChunkResponse])
async def list_document_chunks(document_id: str):
    try:
        existing_doc = get_document_info(document_id)
        if not existing_doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
            
        chunks = get_document_chunks(document_id)
        
        response_chunks = []
        for chunk in chunks:
            # Reconstruct ChunkMetadata from payload
            metadata = ChunkMetadata(
                document_id=chunk.get("document_id", ""),
                filename=chunk.get("filename", ""),
                file_type=chunk.get("file_type", ""),
                page=chunk.get("page"),
                chunk_index=chunk.get("chunk_index", 0),
                char_start=chunk.get("char_start", 0),
                char_end=chunk.get("char_end", 0)
            )
            response_chunks.append(ChunkResponse(
                text=chunk.get("text", ""),
                metadata=metadata
            ))
            
        return response_chunks
    except VectorStorageError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
