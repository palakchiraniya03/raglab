from pydantic import BaseModel
from typing import List, Optional

class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    file_type: str
    page: Optional[int]
    chunk_index: int
    char_start: int
    char_end: int

class ChunkResponse(BaseModel):
    text: str
    metadata: ChunkMetadata

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    total_characters: int
    total_chunks: int
    embedded_chunks: int
    collection: str
    chunks: List[ChunkResponse]

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5

class RetrievalResult(BaseModel):
    text: str
    score: float
    metadata: dict

class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResult]

class GenerationRequest(BaseModel):
    prompt: str

class GenerationResponse(BaseModel):
    text: str

class RAGResponse(BaseModel):
    answer: str
    sources: List[RetrievalResult]
