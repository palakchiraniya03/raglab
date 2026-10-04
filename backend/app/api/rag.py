from fastapi import APIRouter, HTTPException, status
from app.schemas import RetrievalRequest, RAGResponse, RetrievalResult
from app.services.retrieval import search_chunks, RetrievalError
from app.services.generation import generate_text, GenerationError

router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/ask", response_model=RAGResponse)
async def ask_question(request: RetrievalRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")
        
    if request.top_k <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="top_k must be a positive integer")

    # 1. Retrieve chunks
    try:
        retrieved_chunks = await search_chunks(query=request.query, top_k=request.top_k)
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
    prompt = f"""You are a document question-answering assistant. The retrieved context is your sole source of truth.

Important Instructions:
- Keep the answer concise.
- If the context contains an explicit definition, use that definition as the basis of the answer.
- Do not replace an explicit definition with a related concept from general knowledge.
- Do not introduce facts, definitions, formulas, or interpretations that are not supported by the context.
- For conceptual questions, prefer the exact mathematical/textual definition present in the context over related examples, applications, algorithms, or code.
- If multiple context chunks contain relevant information, combine them only when they are consistent.
- If the context does not clearly support an answer, say that the information is not available in the provided documents.

Context:
{context_string}

Question:
{request.query}

Answer:"""

    # 5. Generate answer
    try:
        answer = await generate_text(prompt=prompt)
    except GenerationError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Generation failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected generation error: {str(e)}")

    # 6. Return response
    return RAGResponse(
        answer=answer,
        sources=sources
    )
