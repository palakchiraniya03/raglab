from fastapi import APIRouter
from app.schemas import EvaluationResponse
from app.services.evaluation import run_evaluation

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

@router.post("/run", response_model=EvaluationResponse)
async def evaluate_rag():
    return await run_evaluation()
