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
                metadata=chunk["metadata"]
            )
        )

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
    prompt = f"""You are a document question-answering assistant. Answer the user's question using only the provided context. Do not use outside knowledge or invent facts. If the answer cannot be determined from the context, explicitly say that the answer is not available in the provided documents.

Important Instructions:
- Keep answers concise and directly answer the user's question.
- For conceptual or theoretical questions, prefer the actual definition, explanation, mathematical formula, or description from the context.
- If the context contains both a conceptual definition and a programming implementation (like Python or NetworkX code), do NOT treat the function name or code snippet as the definition. Do not copy a function name as the answer when a mathematical or textual definition is provided.

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
