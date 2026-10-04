from fastapi import APIRouter, HTTPException, status
from app.schemas import GenerationRequest, GenerationResponse
from app.services.generation import generate_text, GenerationError

router = APIRouter(prefix="/generation", tags=["generation"])

@router.post("/generate", response_model=GenerationResponse)
async def generate_endpoint(request: GenerationRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt cannot be empty")
        
    try:
        text = await generate_text(request.prompt)
        return GenerationResponse(text=text)
    except GenerationError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {str(e)}")
