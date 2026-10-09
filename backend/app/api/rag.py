import hashlib
from fastapi import APIRouter, HTTPException, status
from app.schemas import RetrievalRequest, RAGResponse, RetrievalResult, ReproducibilityInfo
from app.services.retrieval import search_chunks, RetrievalError
from app.services.generation import generate_text, GenerationError
from app.services.activity_log import log_activity

router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/ask", response_model=RAGResponse)
async def ask_question(request: RetrievalRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")
        
    if request.top_k <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="top_k must be a positive integer")

    # 1. Retrieve chunks
    try:
        retrieved_chunks = await search_chunks(query=request.query, top_k=request.top_k, document_id=request.document_id)
    except RetrievalError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Retrieval failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected retrieval error: {str(e)}")

    # Format sources
    sources = []
    for chunk in retrieved_chunks:
        sources.append(
            RetrievalResult(
                text=chunk["text"],
                score=chunk["score"],
                semantic_score=chunk.get("semantic_score"),
                lexical_boost=chunk.get("lexical_boost"),
                final_score=chunk.get("final_score"),
                selected=chunk.get("selected"),
                metadata=chunk["metadata"]
            )
        )
        
    # Limit context chunks
    from app.config import settings
    sources = sources[:settings.RAG_MAX_CONTEXT_CHUNKS]

    # 2. Check if context is found
    if not sources:
        return RAGResponse(
            answer="I'm sorry, but I could not find any relevant information in the provided documents to answer your question.",
            sources=[]
        )

    # 3. Build context string
    context_parts = []
    for i, source in enumerate(sources, 1):
        filename = source.metadata.get("filename", "Unknown")
        page = source.metadata.get("page")
        page_str = f"\nPage: {page}" if page is not None else ""
        context_parts.append(
            f"[Source {i}]\nFilename: {filename}{page_str}\nContent:\n{source.text}\n"
        )
    
    context_string = "\n".join(context_parts)

    # 4. Build prompt
    if request.operation == "compare":
        prompt = f"""
    Compare the information from the provided document context.

    Context:
    {context_string}

    Topic to compare:
    {request.query}

    Instructions:
    - Use the context as the source of truth.
    - Explicitly compare information from the different source documents.
    - Highlight important similarities and differences when the context supports them.
    - Identify which source document supports each important point.
    - When answering, you MUST cite the specific sources using the format [Source N] at the end of each relevant sentence or claim. For example: "The matrix is diagonal [Source 1]." Do not hallucinate citations.
    - Do not use outside knowledge or invent information.
    - If the provided context does not contain enough information for a meaningful comparison, clearly state that.
    - Keep the answer concise.

    Answer:
    """
    else:
        prompt = f"""
    Answer the user's question using the provided document context.

    Context:
    {context_string}

    Question:
    {request.query}

    Instructions:
    - Use the context as the source of truth.
    - Synthesize information from multiple sources if the question requires it.
    - When answering, you MUST cite the specific sources used to support your claims using the format [Source N] at the end of each relevant sentence. For example: "The matrix is diagonal [Source 1]." Do not hallucinate citations.
    - For conceptual/definition questions, answer with the conceptual definition from the document, not an implementation function or code snippet.
    - If the context contains both a prose definition and implementation code, prefer the prose definition unless explicitly asked for code.
    - Include the important defining details explicitly stated in the context, such as formulas/values (e.g., 1 and 0), named terminology (e.g., Fiedler Value), minimum/maximum conditions, and relevant start/target concepts.
    - Do not merely copy a function name as the answer to a conceptual question.
    - Do not use outside knowledge or invent details.
    - If the context genuinely does not contain enough information to answer the question, clearly state that the answer is not available in the provided documents.
    - Keep the answer concise, preferably 1-2 sentences.

    Answer:
    """

    # 5. Generate answer
    try:
        answer = await generate_text(prompt=prompt)
    except GenerationError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Generation failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected generation error: {str(e)}")

    # 6. Return response
    prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    
    try:
        log_activity(
            event_type="QUERY",
            document_id=None,
            details={
                "query": request.query,
                "top_k": request.top_k,
                "selected_source_count": len(sources),
                "answer_generated": True,
                "prompt_hash": prompt_hash
            }
        )
    except Exception as e:
        print(f"Audit log failed: {e}")

    reproducibility = ReproducibilityInfo(
        prompt=prompt,
        model=settings.GENERATION_MODEL,
        temperature=0.0
    )
        
    return RAGResponse(
        answer=answer,
        sources=sources,
        reproducibility=reproducibility
    )
