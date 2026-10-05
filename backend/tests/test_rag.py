import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.main import app
from app.services.retrieval import RetrievalError
from app.services.generation import GenerationError

client = TestClient(app)

@patch("app.api.rag.log_activity")
@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_success(mock_search_chunks, mock_generate_text, mock_log_activity):
    mock_search_chunks.return_value = [
        {
            "text": "The Laplacian matrix is L = D - A.",
            "score": 0.95,
            "metadata": {
                "document_id": "test_doc_1",
                "filename": "math.txt",
                "file_type": "txt",
                "page": None,
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 45
            }
        },
        {
            "text": "It has properties related to graph connectivity.",
            "score": 0.85,
            "metadata": {
                "document_id": "test_doc_2",
                "filename": "graph.pdf",
                "file_type": "pdf",
                "page": 2,
                "chunk_index": 1,
                "char_start": 100,
                "char_end": 146
            }
        }
    ]

    mock_generate_text.return_value = "The Laplacian matrix is L = D - A and it relates to graph connectivity."

    response = client.post(
        "/api/rag/ask",
        json={"query": "What is the Laplacian matrix?", "top_k": 2}
    )

    assert response.status_code == 200
    data = response.json()

    from app.config import settings
    expected_count = min(2, settings.RAG_MAX_CONTEXT_CHUNKS)

    assert data["answer"] == "The Laplacian matrix is L = D - A and it relates to graph connectivity."
    assert len(data["sources"]) == expected_count
    assert data["sources"][0]["text"] == "The Laplacian matrix is L = D - A."

    mock_search_chunks.assert_called_once_with(query="What is the Laplacian matrix?", top_k=2, document_id=None)
    mock_generate_text.assert_called_once()
    
    import hashlib
    from app.config import settings
    expected_prompt = mock_generate_text.call_args[1]["prompt"]
    assert "reproducibility" in data
    assert data["reproducibility"]["prompt"] == expected_prompt
    assert data["reproducibility"]["model"] == settings.GENERATION_MODEL
    assert data["reproducibility"]["temperature"] == 0.0

    # Verify prompt construction
    prompt = mock_generate_text.call_args[1]["prompt"]
    assert "[Source 1]" in prompt
    assert "math.txt" in prompt
    if expected_count > 1:
        assert "[Source 2]" in prompt
        assert "graph.pdf" in prompt
        assert "Page: 2" in prompt
    assert "What is the Laplacian matrix?" in prompt
    assert "using the provided document context" in prompt

    # Verify log_activity prompt_hash
    mock_log_activity.assert_called_once()
    log_args = mock_log_activity.call_args[1]
    assert "prompt_hash" in log_args["details"]
    assert log_args["details"]["prompt_hash"] == hashlib.sha256(expected_prompt.encode('utf-8')).hexdigest()

@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_cross_document_context_building(mock_search_chunks, mock_generate_text):
    mock_search_chunks.return_value = [
        {
            "text": "The CEO of RAGLab is Jane Doe.",
            "score": 0.95,
            "metadata": {"document_id": "doc_1", "filename": "ceo_info.txt", "file_type": "txt", "page": None, "chunk_index": 0, "char_start": 0, "char_end": 30}
        },
        {
            "text": "Jane Doe's favorite language is Python.",
            "score": 0.90,
            "metadata": {"document_id": "doc_2", "filename": "personal.txt", "file_type": "txt", "page": 1, "chunk_index": 0, "char_start": 0, "char_end": 39}
        },
        {
            "text": "RAGLab uses FastAPI.",
            "score": 0.85,
            "metadata": {"document_id": "doc_3", "filename": "tech.md", "file_type": "md", "page": None, "chunk_index": 0, "char_start": 0, "char_end": 20}
        }
    ]

    mock_generate_text.return_value = "Jane Doe's favorite programming language is Python."

    response = client.post(
        "/api/rag/ask",
        json={"query": "What is the favorite programming language of RAGLab's CEO?", "top_k": 3}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 3

    prompt = mock_generate_text.call_args[1]["prompt"]
    
    # Verify all three chunks are in the generation prompt
    assert "[Source 1]" in prompt
    assert "ceo_info.txt" in prompt
    assert "The CEO of RAGLab is Jane Doe." in prompt
    
    assert "[Source 2]" in prompt
    assert "personal.txt" in prompt
    assert "Page: 1" in prompt
    assert "Jane Doe's favorite language is Python." in prompt
    
    assert "[Source 3]" in prompt
    assert "tech.md" in prompt
    
    # Verify new instruction is present
    assert "Synthesize information from multiple sources if the question requires it." in prompt

@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_compare_operation(mock_search_chunks, mock_generate_text):
    mock_search_chunks.return_value = [
        {
            "text": "A virus is not a living organism.",
            "score": 0.95,
            "metadata": {"document_id": "doc_1", "filename": "biology.txt", "file_type": "txt", "page": None, "chunk_index": 0, "char_start": 0, "char_end": 30}
        },
        {
            "text": "A virus is a complex living organism.",
            "score": 0.90,
            "metadata": {"document_id": "doc_2", "filename": "alt_science.txt", "file_type": "txt", "page": 1, "chunk_index": 0, "char_start": 0, "char_end": 39}
        }
    ]

    mock_generate_text.return_value = "biology.txt says a virus is not living, while alt_science.txt says it is."

    response = client.post(
        "/api/rag/ask",
        json={"query": "Is a virus living?", "operation": "compare"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 2

    prompt = mock_generate_text.call_args[1]["prompt"]
    
    assert "[Source 1]" in prompt
    assert "biology.txt" in prompt
    assert "[Source 2]" in prompt
    assert "alt_science.txt" in prompt
    
    # Verify compare specific instructions
    assert "Compare the information from the provided document context." in prompt
    assert "Explicitly compare information from the different source documents." in prompt
    assert "Topic to compare:" in prompt
    
    # Verify it does NOT contain standard QA instructions
    assert "For conceptual/definition questions" not in prompt

    assert "reproducibility" in data
    assert data["reproducibility"]["prompt"] == prompt
    assert data["reproducibility"]["temperature"] == 0.0

@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_no_context(mock_search_chunks, mock_generate_text):
    mock_search_chunks.return_value = []

    response = client.post(
        "/api/rag/ask",
        json={"query": "What is the meaning of life?"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "could not find any relevant information" in data["answer"].lower()
    assert len(data["sources"]) == 0

    mock_search_chunks.assert_called_once()
    mock_generate_text.assert_not_called()


@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_retrieval_failure(mock_search_chunks):
    mock_search_chunks.side_effect = RetrievalError("Database connection lost")

    response = client.post(
        "/api/rag/ask",
        json={"query": "test"}
    )

    assert response.status_code == 500
    assert "Retrieval failed" in response.json()["detail"]


@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_generation_failure(mock_search_chunks, mock_generate_text):
    mock_search_chunks.return_value = [
        {
            "text": "Some text",
            "score": 0.9,
            "metadata": {
                "document_id": "1", "filename": "1.txt", "file_type": "txt",
                "page": None, "chunk_index": 0, "char_start": 0, "char_end": 10
            }
        }
    ]
    mock_generate_text.side_effect = GenerationError("Ollama timeout")

    response = client.post(
        "/api/rag/ask",
        json={"query": "test"}
    )

    assert response.status_code == 502
    assert "Generation failed" in response.json()["detail"]

def test_rag_empty_query():
    response = client.post(
        "/api/rag/ask",
        json={"query": "", "top_k": 5}
    )
    assert response.status_code == 400

def test_rag_invalid_top_k():
    response = client.post(
        "/api/rag/ask",
        json={"query": "valid query", "top_k": 0}
    )
    assert response.status_code == 400

@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_max_context_chunks(mock_search_chunks, mock_generate_text):
    # Mock returning 5 relevant chunks
    mock_search_chunks.return_value = [
        {"text": f"Chunk {i}", "score": 0.9, "metadata": {"document_id": "1", "filename": "1.txt", "file_type": "txt", "page": None, "chunk_index": i, "char_start": 0, "char_end": 10}}
        for i in range(5)
    ]

    mock_generate_text.return_value = "Mocked answer"

    response = client.post(
        "/api/rag/ask",
        json={"query": "test query", "top_k": 5}
    )

    assert response.status_code == 200
    data = response.json()

    from app.config import settings
    # The API should limit to RAG_MAX_CONTEXT_CHUNKS (which defaults to 3)
    expected_limit = settings.RAG_MAX_CONTEXT_CHUNKS

    assert len(data["sources"]) == expected_limit
    assert data["sources"][0]["text"] == "Chunk 0"
    if expected_limit > 1:
        assert data["sources"][1]["text"] == "Chunk 1"
    if expected_limit > 2:
        assert data["sources"][2]["text"] == "Chunk 2"

    prompt = mock_generate_text.call_args[1]["prompt"]
    for i in range(expected_limit):
        assert f"Chunk {i}" in prompt
    for i in range(expected_limit, 5):
        assert f"Chunk {i}" not in prompt

@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_fewer_than_max_chunks(mock_search_chunks, mock_generate_text):
    # Mock returning 2 chunks (less than max 3)
    mock_search_chunks.return_value = [
        {"text": f"Chunk {i}", "score": 0.9, "metadata": {"document_id": "1", "filename": "1.txt", "file_type": "txt", "page": None, "chunk_index": i, "char_start": 0, "char_end": 10}}
        for i in range(2)
    ]

    mock_generate_text.return_value = "Mocked answer"

    response = client.post(
        "/api/rag/ask",
        json={"query": "test query", "top_k": 5}
    )

    assert response.status_code == 200
    data = response.json()
    from app.config import settings
    assert len(data["sources"]) == min(2, settings.RAG_MAX_CONTEXT_CHUNKS)

@patch("app.api.rag.generate_text", new_callable=AsyncMock)
@patch("app.api.rag.search_chunks", new_callable=AsyncMock)
def test_rag_fiedler_diagnostics(mock_search_chunks, mock_generate_text):
    mock_search_chunks.return_value = [
        {
            "text": "The Fiedler value is...",
            "score": 0.52,
            "semantic_score": 0.45,
            "lexical_boost": 0.07,
            "final_score": 0.52,
            "selected": True,
            "metadata": {
                "document_id": "1", "filename": "1.txt", "file_type": "txt",
                "page": None, "chunk_index": 0, "char_start": 0, "char_end": 10
            }
        }
    ]

    mock_generate_text.return_value = "Mocked answer"

    response = client.post(
        "/api/rag/ask",
        json={"query": "What is the Fiedler value?", "top_k": 5}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 1

    source = data["sources"][0]
    assert abs(source["semantic_score"] - 0.45) < 1e-6
    assert abs(source["lexical_boost"] - 0.07) < 1e-6
    assert abs(source["final_score"] - 0.52) < 1e-6
