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

class DocumentItem(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    total_characters: int
    total_chunks: int
    embedded_chunks: int
    collection: str
    chunks: List[ChunkResponse] = []
    already_indexed: bool = False

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: Optional[str] = None
    operation: str = "ask"

class RetrievalResult(BaseModel):
    text: str
    score: float
    semantic_score: Optional[float] = None
    lexical_boost: Optional[float] = None
    final_score: Optional[float] = None
    selected: Optional[bool] = None
    metadata: dict

class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResult]

class GenerationRequest(BaseModel):
    prompt: str

class GenerationResponse(BaseModel):
    text: str

class ReproducibilityInfo(BaseModel):
    prompt: str
    model: str
    temperature: float

class RAGResponse(BaseModel):
    answer: str
    sources: List[RetrievalResult]
    reproducibility: Optional[ReproducibilityInfo] = None


class ChunkEvalInfo(BaseModel):
    rank: int
    filename: str
    metadata: dict
    is_relevant: bool

class EvaluationCaseResult(BaseModel):
    id: str
    category: str
    question: str
    answerable: bool
    answer: str
    expected_terms: List[str]
    missing_answer_terms: List[str]
    missing_source_terms: List[str]
    sources_count: int
    has_sources: bool
    answer_passed: bool
    sources_passed: bool
    latency: float
    diagnosis: str
    retrieval_hit: Optional[bool] = None
    precision_at_k: Optional[float] = None
    mrr: Optional[float] = None
    reproducibility: Optional[ReproducibilityInfo] = None
    retrieved_chunks_info: Optional[List[ChunkEvalInfo]] = None

class EvaluationSummary(BaseModel):
    total_questions: int
    answerable_questions: int
    unanswerable_questions: int
    retrieval_success_count: int
    answer_term_pass_count: int
    refusal_success_count: int
    average_latency: float
    avg_precision_at_k: float = 0.0
    avg_mrr: float = 0.0
    retrieval_hit_rate: float = 0.0

class EvaluationResponse(BaseModel):
    summary: EvaluationSummary
    results: List[EvaluationCaseResult]
