import json
import urllib.request
import urllib.error
import time
import os
import sys
import unicodedata
import re

# Force stdout to be utf-8 to prevent cp1252 crash on Windows
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

def normalize_text(text):
    text = unicodedata.normalize('NFKC', text).lower().strip()
    return re.sub(r'\s+', ' ', text)

def evaluate():
    questions_path = os.path.join(os.path.dirname(__file__), 'questions.json')
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
        
    url = "http://127.0.0.1:8000/api/rag/ask"
    headers = {"Content-Type": "application/json"}
    
    total = len(questions)
    answerable_count = 0
    unanswerable_count = 0
    expected_terms_found_count = 0
    sources_found_count = 0
    total_latency = 0.0
    
    likely_generation_failures = 0
    likely_retrieval_failures = 0
    failed_cases = []
    
    for q in questions:
        print(f"Evaluating {q['id']}: {q['question']}")
        req_body = json.dumps({"query": q['question']}).encode('utf-8')
        req = urllib.request.Request(url, data=req_body, headers=headers)
        
        start_time = time.time()
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            print(f"  Request failed: {e}")
            continue
            
        latency = time.time() - start_time
        total_latency += latency
        
        answer = res_data.get('answer', '')
        sources = res_data.get('sources', [])
        sources_count = len(sources)
        has_sources = sources_count > 0
        
        norm_answer = normalize_text(answer)
        
        combined_sources_text = " ".join([s.get("text", "") for s in sources])
        norm_sources = normalize_text(combined_sources_text)
        
        if q['answerable']:
            answerable_count += 1
            if has_sources:
                sources_found_count += 1
                
            missing_terms_answer = []
            missing_terms_sources = []
            
            for term in q['expected_terms']:
                norm_term = normalize_text(term)
                if norm_term not in norm_answer:
                    missing_terms_answer.append(term)
                if norm_term not in norm_sources:
                    missing_terms_sources.append(term)
                    
            if not missing_terms_answer:
                expected_terms_found_count += 1
                print(f"  [PASS] Terms found in answer in {latency:.2f}s")
            else:
                print(f"  [FAIL] Missing from answer: {missing_terms_answer} in {latency:.2f}s")
                
                # Diagnostics
                sources_have_terms = len(missing_terms_sources) == 0
                if sources_have_terms:
                    diagnosis = "Likely generation failure"
                    likely_generation_failures += 1
                else:
                    diagnosis = "Likely retrieval failure"
                    likely_retrieval_failures += 1
                
                failed_cases.append({
                    'id': q['id'],
                    'question': q['question'],
                    'answer': answer,
                    'missing_answer': missing_terms_answer,
                    'missing_sources': missing_terms_sources,
                    'sources_count': sources_count,
                    'sources_have_terms': sources_have_terms,
                    'diagnosis': diagnosis
                })
        else:
            unanswerable_count += 1
            unanswerable_indicators = ['not available', 'don\'t know', 'do not know', 'sorry', 'cannot answer', 'not mention', 'no information']
            if any(ind in norm_answer for ind in unanswerable_indicators):
                print(f"  [PASS] Correctly identified as unanswerable in {latency:.2f}s")
            else:
                print(f"  [FAIL] Did not properly indicate unanswerable in {latency:.2f}s")
                failed_cases.append({
                    'id': q['id'],
                    'question': q['question'],
                    'answer': answer,
                    'missing_answer': ["Unanswerable fallback missing"],
                    'missing_sources': [],
                    'sources_count': sources_count,
                    'sources_have_terms': False,
                    'diagnosis': "Unanswerable fallback missing"
                })
                
    print("\n--- Final Summary ---")
    print(f"Total questions: {total}")
    print(f"Answerable questions: {answerable_count}")
    print(f"Unanswerable questions: {unanswerable_count}")
    print(f"Answerable cases with expected terms found: {expected_terms_found_count}/{answerable_count}")
    print(f"Answerable cases with sources: {sources_found_count}/{answerable_count}")
    avg_latency = total_latency / total if total > 0 else 0
    print(f"Average latency: {avg_latency:.2f}s")
    
    print("\n--- Diagnostics ---")
    print(f"Answerable cases:\n{expected_terms_found_count}/{answerable_count} passed answer-term check")
    
    if likely_generation_failures > 0 or likely_retrieval_failures > 0:
        print(f"\nOf failed answerable cases:")
        print(f"- likely generation failures: {likely_generation_failures}")
        print(f"- likely retrieval failures: {likely_retrieval_failures}")
    
    if failed_cases:
        print("\n--- Failed Cases Details ---")
        for f in failed_cases:
            print(f"ID: {f['id']}")
            print(f"Question: {f['question']}")
            print(f"Answer: {f['answer']}")
            
            if f['id'] != 'q10':
                answer_passed = "YES" if not f['missing_answer'] else "NO"
                sources_terms_present = "YES" if f['sources_have_terms'] else "NO"
                print(f"Answer expected terms found: {answer_passed}")
                print(f"Retrieved sources: {f['sources_count']}")
                print(f"Expected terms present in sources: {sources_terms_present}")
                if not f['sources_have_terms']:
                    print(f"Terms missing from sources: {f['missing_sources']}")
                print(f"Diagnosis: {f['diagnosis']}\n")
            else:
                print(f"Diagnosis: {f['diagnosis']}\n")

if __name__ == '__main__':
    evaluate()
