import json
import os
import asyncio
import sys

from app.config import settings
from app.services.retrieval import search_chunks
from app.services.evaluation import is_chunk_relevant

async def run_experiment():
    print("Lexical Boost Experiment\n")
    
    questions_path = os.path.join(os.path.dirname(__file__), 'questions.json')
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    weights_to_test = [0.00, 0.03, 0.05, 0.07, 0.10]
    original_weight = settings.LEXICAL_BOOST_WEIGHT
    
    summary_rows = []
    
    try:
        for weight in weights_to_test:
            settings.LEXICAL_BOOST_WEIGHT = weight
            
            print(f"--- Weight: {weight:.2f} ---")
            
            sum_precision = 0.0
            sum_mrr = 0.0
            total_hits = 0
            answerable_count = 0
            
            for q in questions:
                if not q.get('answerable', True):
                    continue
                    
                answerable_count += 1
                
                try:
                    raw_chunks = await search_chunks(query=q['question'], top_k=5)
                    chunks = raw_chunks[:settings.RAG_MAX_CONTEXT_CHUNKS]
                    
                    hit_count = 0
                    first_hit_rank = -1
                    
                    for rank_idx, chunk in enumerate(chunks):
                        if is_chunk_relevant(chunk['text'], q):
                            hit_count += 1
                            if first_hit_rank == -1:
                                first_hit_rank = rank_idx + 1
                                
                    retrieval_hit = hit_count > 0
                    precision_at_k = hit_count / len(chunks) if len(chunks) > 0 else 0.0
                    mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0
                    
                    if retrieval_hit:
                        total_hits += 1
                    sum_precision += precision_at_k
                    sum_mrr += mrr
                    
                    print(f"  {q['id']}: hit={retrieval_hit}, P@K={precision_at_k:.2f}, MRR={mrr:.2f}")
                    
                except Exception as e:
                    print(f"  {q['id']}: Error during retrieval - {str(e)}")
            
            hit_rate = (total_hits / answerable_count) * 100 if answerable_count > 0 else 0.0
            avg_precision = sum_precision / answerable_count if answerable_count > 0 else 0.0
            avg_mrr = sum_mrr / answerable_count if answerable_count > 0 else 0.0
            
            summary_rows.append(f"| {weight:.2f}   | {hit_rate:6.1f}%   | {avg_precision:.3f}     | {avg_mrr:.3f} | {answerable_count} |")
            print()
            
    finally:
        settings.LEXICAL_BOOST_WEIGHT = original_weight
        print(f"Restored settings.LEXICAL_BOOST_WEIGHT to {settings.LEXICAL_BOOST_WEIGHT}\n")

    print("| Weight | Hit Rate | Precision@K | MRR   | Questions |")
    print("|--------|----------|-------------|-------|-----------|")
    for row in summary_rows:
        print(row)
    print("\nExperiment completed successfully.")

if __name__ == '__main__':
    asyncio.run(run_experiment())
