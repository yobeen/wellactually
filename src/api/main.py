# src/api/main.py
"""
FastAPI server for LLM repository comparison and originality assessment API.
Enhanced with criteria assessment endpoint.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from wellactually.src.api.requests import ComparisonRequest, OriginalityRequest
from wellactually.src.api.responses import ComparisonResponse, OriginalityResponse, ErrorResponse
from wellactually.src.api.criteria_models import CriteriaRequest, CriteriaResponse
from wellactually.src.api.llm_orchestrator import LLMOrchestrator
from wellactually.src.api.comparison_handler import ComparisonHandler
from wellactually.src.api.originality_handler import OriginalityHandler
from wellactually.src.api.criteria_handler import CriteriaHandler
from wellactually.src.api.settings import APISettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components
llm_orchestrator = None
comparison_handler = None
originality_handler = None
criteria_handler = None
settings = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global llm_orchestrator, comparison_handler, originality_handler, criteria_handler, settings
    
    # Startup
    try:
        logger.info("Initializing LLM API server...")
        
        # Load configuration
        settings = APISettings()
        
        # Initialize core orchestrator
        llm_orchestrator = LLMOrchestrator(settings.llm_config)
        
        # Initialize handlers
        comparison_handler = ComparisonHandler(llm_orchestrator)
        originality_handler = OriginalityHandler(llm_orchestrator)
        criteria_handler = CriteriaHandler(llm_orchestrator)
        
        logger.info("LLM API server startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM API server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM API server...")

# Create FastAPI app
app = FastAPI(
    title="LLM Repository Assessment API",
    description="API for repository comparison, originality assessment, and criteria evaluation using LLMs",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependencies to get handlers
def get_comparison_handler() -> ComparisonHandler:
    """Dependency to get comparison handler."""
    if comparison_handler is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return comparison_handler

def get_originality_handler() -> OriginalityHandler:
    """Dependency to get originality handler."""
    if originality_handler is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return originality_handler

def get_criteria_handler() -> CriteriaHandler:
    """Dependency to get criteria handler."""
    if criteria_handler is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return criteria_handler

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred"
        ).dict()
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "llm-repository-api"}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "LLM Repository Assessment API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "comparison": "/compare",
            "originality": "/assess",
            "criteria": "/criteria"
        }
    }

@app.post("/compare", response_model=ComparisonResponse)
async def compare_repositories(
    request: ComparisonRequest,
    handler: ComparisonHandler = Depends(get_comparison_handler)
) -> ComparisonResponse:
    """
    Compare two repositories for their relative importance.
    
    Args:
        request: Comparison request with repo_a, repo_b, parent, and optional parameters
        
    Returns:
        ComparisonResponse with choice, multiplier, uncertainties, and explanation
    """
    try:
        logger.info(f"Processing comparison: {request.repo_a} vs {request.repo_b} -> {request.parent}")
        
        # Route to appropriate handler based on parent
        if request.parent.lower() == "ethereum":
            response = await handler.handle_l1_comparison(request)
        else:
            response = await handler.handle_l3_comparison(request)
        
        logger.info(f"Comparison complete: choice={response.choice}, multiplier={response.multiplier}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in comparison: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process comparison request")

@app.post("/assess", response_model=OriginalityResponse)
async def assess_originality(
    request: OriginalityRequest,
    handler: OriginalityHandler = Depends(get_originality_handler)
) -> OriginalityResponse:
    """
    Assess repository originality vs dependency reliance.
    
    Args:
        request: Originality request with repo and optional parameters
        
    Returns:
        OriginalityResponse with originality score, uncertainty, and explanation
    """
    try:
        logger.info(f"Processing originality assessment: {request.repo}")
        
        response = await handler.handle_originality_assessment(request)
        
        logger.info(f"Originality assessment complete: score={response.originality}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in originality assessment: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing originality assessment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process originality assessment")

@app.post("/criteria", response_model=CriteriaResponse)
async def assess_criteria(
    request: CriteriaRequest,
    handler: CriteriaHandler = Depends(get_criteria_handler)
) -> CriteriaResponse:
    """
    Assess repository against 11 importance criteria for Ethereum ecosystem.
    
    Args:
        request: Criteria request with repo and optional parameters
        
    Returns:
        CriteriaResponse with detailed criteria scores, weights, reasoning, and uncertainties
    """
    try:
        logger.info(f"Processing criteria assessment: {request.repo}")
        
        response = await handler.handle_criteria_assessment(request)
        
        logger.info(f"Criteria assessment complete: target_score={response.target_score:.2f}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in criteria assessment: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing criteria assessment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process criteria assessment")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics for monitoring."""
    try:
        if llm_orchestrator and llm_orchestrator.multi_model_engine:
            stats = llm_orchestrator.multi_model_engine.get_cache_stats()
            return {"cache_stats": stats}
        else:
            return {"error": "LLM orchestrator not initialized"}
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cache stats")

if __name__ == "__main__":
    # For development only
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )