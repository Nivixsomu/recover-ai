"""FastAPI application entry point for RecoverAI."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.db.session import init_db
from backend.app.routers import metrics_router, recovery_router

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize database and load resources on startup."""
    init_db()
    yield


app = FastAPI(
    title="RecoverAI — AI Revenue Recovery API",
    description="Intelligent, Policy-Bounded Action-Conditioned Payment Recovery Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for dashboard and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(recovery_router)
app.include_router(metrics_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "healthy"}


# Mount static frontend if available
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
