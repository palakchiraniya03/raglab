import json
import os
import time
import unicodedata
import re
import asyncio
from fastapi import HTTPException
from app.api.rag import ask_question
from app.schemas import RetrievalRequest, EvaluationResponse, EvaluationSummary, EvaluationCaseResult, ChunkEvalInfo

EVALUATION_CASE_TIMEOUT_SECONDS = 120

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).lower().strip()
    return re.sub(r'\s+', ' ', text)

def is_chunk_relevant(chunk_text: str, q: dict) -> bool:
    norm_text = normalize_text(chunk_text)
    expected_terms = q.get('expected_terms', [])
    acceptable_terms = q.get('acceptable_terms', {})

    if not expected_terms:
        return False

    # The first expected term is the primary concept (e.g. "adjacency matrix")
    primary_term = expected_terms[0]
    primary_alts = [primary_term] + acceptable_terms.get(primary_term, [])

    return any(normalize_text(alt) in norm_text for alt in primary_alts)


def validate_citations(answer: str, sources: list, expected_terms: list) -> dict:
    import re
    citation_regex = r'\[Source\s+(\d+)\]'
    citations = list(re.finditer(citation_regex, answer))

    if not citations:
        return {"valid": False, "reason": "Missing citations", "invalid_indices": [], "unsupported": []}

    invalid_indices = []
    unsupported = []

    for match in citations:
        source_idx = int(match.group(1)) - 1
        if source_idx < 0 or source_idx >= len(sources):
            invalid_indices.append(match.group(0))
            continue

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

async def run_evaluation() -> EvaluationResponse:
    questions_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'evaluation', 'questions.json')
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    total = len(questions)
    answerable_count = 0
    unanswerable_count = 0
    expected_terms_found_count = 0
    sources_found_count = 0
    total_latency = 0.0

    sum_precision = 0.0
    sum_mrr = 0.0
    total_retrieval_hits = 0

    results = []

    for q in questions:
        start_time = time.time()
        reproducibility = None

        try:
            req = RetrievalRequest(query=q['question'], top_k=5)
            res_data = await asyncio.wait_for(
                ask_question(req),
                timeout=EVALUATION_CASE_TIMEOUT_SECONDS
            )
            answer = res_data.answer
            sources = res_data.sources
            reproducibility = res_data.reproducibility
        except asyncio.TimeoutError:
            answer = "Evaluation timed out after 30 seconds."
            sources = []
        except Exception as e:
            answer = f"Error: {str(e)}"
            sources = []

        latency = time.time() - start_time
        total_latency += latency

        sources_count = len(sources)
        has_sources = sources_count > 0

        norm_answer = normalize_text(answer)
        combined_sources_text = " ".join([s.text for s in sources])
        norm_sources = normalize_text(combined_sources_text)

        missing_answer_terms = []
        missing_source_terms = []
        answer_passed = False
        sources_passed = False
        diagnosis = "Pass"

        retrieval_hit = None
        precision_at_k = None
        mrr = None
        retrieved_chunks_info = None

        if q['answerable']:
            answerable_count += 1
            if has_sources:
                sources_found_count += 1

            acceptable_terms = q.get('acceptable_terms', {})
            for term in q['expected_terms']:
                alternatives = [term]
                if term in acceptable_terms:
                    alternatives.extend(acceptable_terms[term])

                # Check answer
                term_found_in_answer = False
                for alt in alternatives:
                    norm_alt = normalize_text(alt)
                    if norm_alt in norm_answer:
                        # Check for negation within the 30 characters preceding the term
                        matches = list(re.finditer(re.escape(norm_alt), norm_answer))
                        any_valid = False
                        for match in matches:
                            preceding = norm_answer[max(0, match.start() - 30):match.start()]
                            # If no negation word is in the preceding window
                            if not re.search(r'\b(not|never|no|isn\'t|doesn\'t|aren\'t|without)\b', preceding):
                                any_valid = True
                                break
                        if any_valid:
                            term_found_in_answer = True
                            break

                if not term_found_in_answer:
                    missing_answer_terms.append(term)

                # Check sources
                if not any(normalize_text(alt) in norm_sources for alt in alternatives):
                    missing_source_terms.append(term)

            answer_passed = len(missing_answer_terms) == 0
            sources_passed = len(missing_source_terms) == 0

            if answer_passed:
                expected_terms_found_count += 1
                diagnosis = "Pass"
            else:
                if sources_passed:
                    diagnosis = "Likely generation failure"
                else:
                    diagnosis = "Likely retrieval failure"

            # Calculate chunk-level metrics
            retrieved_chunks_info = []
            first_hit_rank = -1
            hit_count = 0

            for rank_idx, chunk in enumerate(sources):
                is_rel = is_chunk_relevant(chunk.text, q)
                if is_rel:
                    hit_count += 1
                    if first_hit_rank == -1:
                        first_hit_rank = rank_idx + 1

                retrieved_chunks_info.append(ChunkEvalInfo(
                    rank=rank_idx + 1,
                    filename=chunk.metadata.get("filename", "unknown"),
                    metadata=chunk.metadata,
                    is_relevant=is_rel
                ))

            retrieval_hit = hit_count > 0
            precision_at_k = float(hit_count) / sources_count if sources_count > 0 else 0.0
            mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0

            if retrieval_hit:
                total_retrieval_hits += 1
            sum_precision += precision_at_k
            sum_mrr += mrr
        else:
            unanswerable_count += 1
            unanswerable_indicators = ['not available', 'don\'t know', 'do not know', 'sorry', 'cannot answer', 'not mention', 'no information']
            answer_passed = any(ind in norm_answer for ind in unanswerable_indicators)
            sources_passed = True

            if answer_passed:
                diagnosis = "Pass"
            else:
                missing_answer_terms = ["Unanswerable fallback missing"]
                diagnosis = "Unanswerable fallback missing"

        results.append(EvaluationCaseResult(
            id=q['id'],
            category=q.get('category', 'unknown'),
            question=q['question'],
            answerable=q['answerable'],
            answer=answer,
            expected_terms=q['expected_terms'],
            missing_answer_terms=missing_answer_terms,
            missing_source_terms=missing_source_terms,
            sources_count=sources_count,
            has_sources=has_sources,
            answer_passed=answer_passed,
            sources_passed=sources_passed,
            latency=latency,
            diagnosis=diagnosis,
            retrieval_hit=retrieval_hit,
            precision_at_k=precision_at_k,
            mrr=mrr,
            reproducibility=reproducibility,
            retrieved_chunks_info=retrieved_chunks_info
        ))

    avg_latency = total_latency / total if total > 0 else 0
    refusal_success_count = sum(1 for r in results if not r.answerable and r.answer_passed)

    avg_precision = sum_precision / answerable_count if answerable_count > 0 else 0.0
    avg_mrr = sum_mrr / answerable_count if answerable_count > 0 else 0.0
    hit_rate = float(total_retrieval_hits) / answerable_count if answerable_count > 0 else 0.0

    summary = EvaluationSummary(
        total_questions=total,
        answerable_questions=answerable_count,
        unanswerable_questions=unanswerable_count,
        retrieval_success_count=sources_found_count,
        answer_term_pass_count=expected_terms_found_count,
        refusal_success_count=refusal_success_count,
        average_latency=avg_latency,
        avg_precision_at_k=avg_precision,
        avg_mrr=avg_mrr,
        retrieval_hit_rate=hit_rate
    )

    return EvaluationResponse(summary=summary, results=results)
