import json
import os
import time
import unicodedata
import re
import asyncio
from fastapi import HTTPException
from app.api.rag import ask_question
from app.schemas import RetrievalRequest, EvaluationResponse, EvaluationSummary, EvaluationCaseResult

EVALUATION_CASE_TIMEOUT_SECONDS = 30

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).lower().strip()
    return re.sub(r'\s+', ' ', text)

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

    results = []

    for q in questions:
        start_time = time.time()

        try:
            req = RetrievalRequest(query=q['question'], top_k=5)
            res_data = await asyncio.wait_for(
                ask_question(req),
                timeout=EVALUATION_CASE_TIMEOUT_SECONDS
            )
            answer = res_data.answer
            sources = res_data.sources
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
                if not any(normalize_text(alt) in norm_answer for alt in alternatives):
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
            diagnosis=diagnosis
        ))

    avg_latency = total_latency / total if total > 0 else 0
    refusal_success_count = sum(1 for r in results if not r.answerable and r.answer_passed)

    summary = EvaluationSummary(
        total_questions=total,
        answerable_questions=answerable_count,
        unanswerable_questions=unanswerable_count,
        retrieval_success_count=sources_found_count,
        answer_term_pass_count=expected_terms_found_count,
        refusal_success_count=refusal_success_count,
        average_latency=avg_latency
    )

    return EvaluationResponse(summary=summary, results=results)
