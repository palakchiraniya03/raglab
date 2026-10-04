from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
import hashlib
import os
from typing import Optional

from app.schemas import IngestResponse, ChunkResponse, ChunkMetadata
from app.services.parsing import parse_document, DocumentParsingError
from app.services.chunking import chunk_text

router = APIRouter(prefix="/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

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

    return IngestResponse(
        document_id=document_id,
        filename=file.filename,
        file_type=ext[1:],
        total_characters=total_characters,
        total_chunks=len(chunks_response),
        chunks=chunks_response
    )
