from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(f"{settings.API_PREFIX}/health")
async def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}

@app.get("/")
async def root():
    return {"message": "Welcome to RAGLab API. Navigate to /api/health for health check."}
