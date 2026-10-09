import pytest
from app.schemas import RetrievalResult
import re
from app.services.evaluation import normalize_text

def validate_citations(answer: str, sources: list[RetrievalResult], expected_terms: list[str]) -> dict:
    citation_regex = r'\[Source\s+(\d+)\]'
    citations = list(re.finditer(citation_regex, answer))

    if not citations:
        return {"valid": False, "reason": "Missing citations", "invalid_indices": [], "unsupported": []}

    invalid_indices = []
    unsupported = []

    # Keep track of multiple citations
    for match in citations:
        source_idx = int(match.group(1)) - 1

        if source_idx < 0 or source_idx >= len(sources):
            invalid_indices.append(match.group(0))
            continue

        # Syntactic heuristic for unsupported claims
        preceding = answer[:match.start()].split('.')[-1]
        chunk_text = normalize_text(sources[source_idx].text)
        sentence_norm = normalize_text(preceding)

        words = [w for w in sentence_norm.split() if len(w) > 4]
        overlap = [w for w in words if w in chunk_text]

        expected_overlap = False
        for term in expected_terms:
            if normalize_text(term) in sentence_norm and normalize_text(term) in chunk_text:
                expected_overlap = True

        if not expected_overlap and len(words) > 0 and len(overlap) == 0:
            unsupported.append(match.group(0))

    is_valid = len(invalid_indices) == 0 and len(unsupported) == 0
    return {
        "valid": is_valid,
        "reason": "Valid" if is_valid else "Invalid",
        "invalid_indices": invalid_indices,
        "unsupported": unsupported,
        "citation_count": len(citations)
    }

def test_valid_citations():
    answer = "The degree matrix is diagonal [Source 1]."
    sources = [RetrievalResult(text="The degree matrix D is a diagonal matrix.", score=0.9, metadata={})]
    res = validate_citations(answer, sources, ["degree matrix", "diagonal matrix"])
    assert res["valid"] == True
    assert res["citation_count"] == 1

def test_missing_citations():
    answer = "The degree matrix is diagonal."
    sources = [RetrievalResult(text="The degree matrix D is a diagonal matrix.", score=0.9, metadata={})]
    res = validate_citations(answer, sources, ["degree matrix"])
    assert res["valid"] == False
    assert res["reason"] == "Missing citations"

def test_invalid_source_indices():
    answer = "The degree matrix is diagonal [Source 2]."
    sources = [RetrievalResult(text="The degree matrix D is a diagonal matrix.", score=0.9, metadata={})]
    res = validate_citations(answer, sources, ["degree matrix"])
    assert res["valid"] == False
    assert "[Source 2]" in res["invalid_indices"]

def test_multiple_citations():
    answer = "The matrix is diagonal [Source 1]. It also represents connections [Source 2]."
    sources = [
        RetrievalResult(text="The degree matrix is diagonal.", score=0.9, metadata={}),
        RetrievalResult(text="Adjacency represents connections.", score=0.8, metadata={})
    ]
    res = validate_citations(answer, sources, ["diagonal", "connections"])
    assert res["valid"] == True
    assert res["citation_count"] == 2

def test_unsupported_claim():
    # The claim uses words not in the source text
    answer = "Elephants are heavy [Source 1]."
    sources = [RetrievalResult(text="The degree matrix is diagonal.", score=0.9, metadata={})]
    res = validate_citations(answer, sources, [])
    assert res["valid"] == False
    assert "[Source 1]" in res["unsupported"]

def test_citation_numbering_filtered_reordered():
    # Simulate ask_question logic
    from app.api.rag import router
    # We just test the assumption that [Source 1] corresponds to sources[0]
    # after filtering
    original_retrieved = [
        RetrievalResult(text="Irrelevant text", score=0.9, metadata={}),
        RetrievalResult(text="The degree matrix is diagonal", score=0.8, metadata={}),
        RetrievalResult(text="More text", score=0.7, metadata={}),
        RetrievalResult(text="Even more", score=0.6, metadata={}),
    ]

    # Filter to top 3 (like settings.RAG_MAX_CONTEXT_CHUNKS = 3)
    filtered_sources = original_retrieved[:3]

    answer = "The degree matrix is diagonal [Source 2]."
    res = validate_citations(answer, filtered_sources, ["degree matrix"])
    # Should be valid because filtered_sources[1] (which is Source 2) contains the support
    assert res["valid"] == True
