from fastapi import APIRouter, HTTPException, status
from app.schemas import RetrievalRequest, RetrievalResponse, RetrievalResult
from app.services.retrieval import search_chunks, RetrievalError
from app.services.vector_storage import VectorStorageError

router = APIRouter(prefix="/retrieval", tags=["retrieval"])

@router.post("/search", response_model=RetrievalResponse)
async def search_documents(request: RetrievalRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")
        
    if request.top_k <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="top_k must be a positive integer")
        
    try:
        results = await search_chunks(query=request.query, top_k=request.top_k, document_id=request.document_id)
    except RetrievalError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {str(e)}")
        
    response_results = []
    for res in results:
        response_results.append(
            RetrievalResult(
                text=res["text"],
                score=res["score"],
                metadata=res["metadata"]
            )
        )
        
    return RetrievalResponse(
        query=request.query,
        results=response_results
    )
