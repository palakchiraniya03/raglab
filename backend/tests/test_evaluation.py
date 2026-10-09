import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_evaluation_run_endpoint(monkeypatch):
    # Mock ask_question so we don't actually hit Qdrant/Ollama
    from app.schemas import RAGResponse, RetrievalResult
    from app.api.rag import ask_question

    async def mock_ask_question(req):
        if "France" in req.query:
            return RAGResponse(
                answer="The information is not available.",
                sources=[]
            )
        elif "adjacency matrix" in req.query.lower():
            # q01: pass - missing "1" and "0" but uses acceptable term "presence"
            return RAGResponse(
                answer="The adjacency matrix indicates presence of an edge or absence.",
                sources=[RetrievalResult(text="The adjacency matrix A is defined by A_i,j = 1 if (v_i, v_j) belongs to E, and 0 otherwise.", score=0.9, metadata={"filename": "test.pdf"})]
            )
        elif "degree matrix" in req.query.lower():
            # q02: fail - missing "diagonal matrix"
            return RAGResponse(
                answer="The degree matrix D is not a diagonal matrix.",
                sources=[RetrievalResult(text="The degree matrix D is a diagonal matrix where D_i,i = deg(v_i).", score=0.9, metadata={"filename": "test.pdf"})]
            )
        elif "unnormalized" in req.query.lower():
            # q03: fail - negates L
            return RAGResponse(
                answer="It is never L.",
                sources=[RetrievalResult(text="L = D - A", score=0.9, metadata={"filename": "test.pdf"})]
            )
        elif "shortest path length" in req.query.lower():
            # q09: pass - uses alternative "minimum number of edges" instead of "distance"
            return RAGResponse(
                answer="Shortest path length calculates the minimum number of edges between the source and target.",
                sources=[RetrievalResult(text="Shortest path length calculates the minimum number of edges between the source and target.", score=0.9, metadata={"filename": "test.pdf"})]
            )

        # default pass for all other answerable
        return RAGResponse(
            answer="dummy adjacency matrix 1 0 degree matrix diagonal matrix deg unnormalized laplacian L D A normalized laplacian algebraic connectivity second smallest eigenvalue Fiedler BFS layer-by-layer source node Queue DFS deep backtracking Stack Recursion shortest path minimum start node target node shortest path length steps distance",
            sources=[RetrievalResult(
                text="dummy adjacency matrix 1 0 degree matrix diagonal matrix deg unnormalized laplacian L D A normalized laplacian algebraic connectivity second smallest eigenvalue Fiedler BFS layer-by-layer source node Queue DFS deep backtracking Stack Recursion shortest path minimum start node target node shortest path length steps distance",
                score=0.9,
                metadata={"filename": "test.pdf"}
            )]
        )

    monkeypatch.setattr('app.services.evaluation.ask_question', mock_ask_question)

    response = client.post("/api/evaluation/run")
    assert response.status_code == 200

    data = response.json()
    assert "summary" in data
    assert "results" in data

    summary = data["summary"]
    assert summary["total_questions"] == 10
    assert summary["answerable_questions"] == 9
    assert summary["unanswerable_questions"] == 1

    results = {r["id"]: r for r in data["results"]}

    # q01 should pass with acceptable terms
    assert results["q01"]["answer_passed"] == True
    assert len(results["q01"]["missing_answer_terms"]) == 0

    # q02 should fail because 'diagonal matrix' is negated
    assert results["q02"]["answer_passed"] == False
    assert "diagonal matrix" in results["q02"]["missing_answer_terms"]

    # q03 should fail because 'L' is negated
    assert results["q03"]["answer_passed"] == False
    assert "L" in results["q03"]["missing_answer_terms"]
    assert results["q09"]["answer_passed"] == True
    assert len(results["q09"]["missing_answer_terms"]) == 0

def test_evaluation_timeout(monkeypatch):
    from app.services import evaluation
    monkeypatch.setattr(evaluation, 'EVALUATION_CASE_TIMEOUT_SECONDS', 0.01)

    from app.schemas import RAGResponse
    import asyncio

    async def mock_ask_question_sleep(req):
        await asyncio.sleep(0.05)
        return RAGResponse(answer="Should not see this", sources=[])

    monkeypatch.setattr('app.services.evaluation.ask_question', mock_ask_question_sleep)

    response = client.post("/api/evaluation/run")
    assert response.status_code == 200

    data = response.json()
    assert len(data["results"]) == 10

    for res in data["results"]:
        assert res["answer"] == "Evaluation timed out after 30 seconds."
        assert res["sources_count"] == 0
        assert res["has_sources"] == False
        if res["answerable"]:
            assert res["retrieval_hit"] == False
            assert res["precision_at_k"] == 0.0
            assert res["mrr"] == 0.0

def test_is_chunk_relevant():
    from app.services.evaluation import is_chunk_relevant
    
    q1 = {
        "expected_terms": ["adjacency matrix", "1", "0"]
    }
    # Has main conceptual term
    assert is_chunk_relevant("The adjacency matrix is nice.", q1) == True
    # Has only supporting term, no main conceptual term
    assert is_chunk_relevant("1 and 0", q1) == False
    
    q2 = {
        "expected_terms": ["BFS", "Queue"]
    }
    # It now strictly checks the primary term (BFS)
    assert is_chunk_relevant("We use a Queue here.", q2) == False
    assert is_chunk_relevant("We use BFS.", q2) == True
    assert is_chunk_relevant("Stack", q2) == False
    
    q3 = {
        "expected_terms": ["shortest path length", "minimum", "distance"],
        "acceptable_terms": {
            "shortest path length": ["distance"]
        }
    }
    # Primary term (shortest path length) missing but acceptable alternative present
    assert is_chunk_relevant("This is the minimum distance.", q3) == True

def test_chunk_level_metrics(monkeypatch):
    from app.schemas import RAGResponse, RetrievalResult
    from app.api.rag import ask_question

    async def mock_ask_question(req):
        if "adjacency matrix" in req.query.lower():
            return RAGResponse(
                answer="dummy",
                sources=[
                    RetrievalResult(text="Just some text 1 0.", score=0.9, metadata={"filename": "test.pdf"}), # Irrelevant
                    RetrievalResult(text="The adjacency matrix is...", score=0.8, metadata={"filename": "test.pdf"}), # Relevant
                    RetrievalResult(text="Another adjacency matrix...", score=0.7, metadata={"filename": "test.pdf"}) # Relevant
                ]
            )
        return RAGResponse(answer="dummy", sources=[])

    monkeypatch.setattr('app.services.evaluation.ask_question', mock_ask_question)
    response = client.post("/api/evaluation/run")
    assert response.status_code == 200
    results = {r["id"]: r for r in response.json()["results"]}
    
    q01 = results["q01"]
    assert q01["retrieval_hit"] == True
    assert q01["precision_at_k"] == 2 / 3
    assert q01["mrr"] == 0.5
    
    chunks = q01["retrieved_chunks_info"]
    assert len(chunks) == 3
    assert chunks[0]["is_relevant"] == False
    assert chunks[1]["is_relevant"] == True
    assert chunks[2]["is_relevant"] == True

