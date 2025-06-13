# src/api/main.py
"""
FastAPI server for LLM repository comparison and originality assessment API.
Enhanced with criteria assessment endpoint and special case comparison support.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import uvicorn

from src.api.core.requests import ComparisonRequest, OriginalityRequest, BatchComparisonRequest
from src.api.core.responses import ComparisonResponse, OriginalityResponse, ErrorResponse, BatchComparisonResponse
from src.api.criteria.criteria_models import CriteriaRequest, CriteriaResponse
from src.api.core.llm_orchestrator import LLMOrchestrator
from src.api.comparison.comparison_handler import ComparisonHandler
from src.api.originality.originality_handler import OriginalityHandler
from src.api.criteria.criteria_handler import CriteriaHandler
from src.api.core.settings import APISettings

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
        
        # Log configuration details
        logger.info(f"Default model configured: {settings.default_model}")
        logger.info(f"Default temperature: {settings.default_temperature}")
        logger.info(f"Cache enabled: {settings.enable_cache}")
        logger.info(f"LLM config path: {settings.llm_config_path}")
        
        # Initialize core orchestrator
        llm_orchestrator = LLMOrchestrator(settings.llm_config, settings.default_model)
        
        # Initialize handlers with special case support
        comparison_handler = ComparisonHandler(llm_orchestrator)
        originality_handler = OriginalityHandler(llm_orchestrator)
        criteria_handler = CriteriaHandler(llm_orchestrator)
        
        # Validate special case data on startup
        logger.info("Validating special case data...")
        validation_results = comparison_handler.validate_special_case_data()
        
        if validation_results.get('valid', False):
            logger.info(f"Special case data valid: {validation_results.get('total_assessments', 0)} assessments available")
        else:
            logger.warning("Special case data validation failed - will fall back to standard LLM comparisons")
            if validation_results.get('errors'):
                for error in validation_results['errors'][:3]:
                    logger.warning(f"  Validation error: {error}")
        
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
    description="API for repository comparison, originality assessment, and criteria evaluation using LLMs with special case support",
    version="1.1.0",
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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with detailed logging."""
    logger.warning(f"Validation error on {request.method} {request.url.path}")
    logger.warning(f"Validation details: {exc.errors()}")
    
    # Try to log the request body if possible
    try:
        body = await request.body()
        if body:
            logger.warning(f"Request body: {body.decode('utf-8')[:500]}...")
    except Exception:
        logger.warning("Could not read request body")
    
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Validation error",
            detail=str(exc)
        ).dict()
    )

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
    """Health check endpoint with special case status."""
    special_case_stats = comparison_handler.get_special_case_stats() if comparison_handler else {}
    
    return {
        "status": "healthy", 
        "service": "llm-repository-api",
        "version": "1.1.0",
        "special_case": {
            "enabled": special_case_stats.get("enabled", False),
            "data_valid": special_case_stats.get("data_valid", False),
            "available_repositories": special_case_stats.get("available_repositories", 0)
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "LLM Repository Assessment API",
        "version": "1.1.0",
        "features": [
            "Repository comparison with special case support",
            "Originality assessment", 
            "Criteria-based evaluation",
            "Automatic fallback to LLM when needed"
        ],
        "endpoints": {
            "health": "/health",
            "comparison": "/compare",
            "batch_comparison": "/compare/batch",
            "originality": "/assess",
            "criteria": "/criteria",
            "special_case_status": "/special-case/status",
            "bulk_cached_comparisons": "/cached-comparisons/bulk",
            "bulk_cached_originality": "/cached-originality/bulk"
        }
    }

@app.post("/compare", response_model=ComparisonResponse)
async def compare_repositories(
    request: ComparisonRequest,
    handler: ComparisonHandler = Depends(get_comparison_handler)
) -> ComparisonResponse:
    """
    Compare two repositories for their relative importance.
    For Level 1 (parent="ethereum"), uses criteria-based comparison when available,
    falls back to LLM-based comparison otherwise.
    
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
        logger.warning(f"Request data: repo={getattr(request, 'repo', 'N/A')}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing originality assessment: {e}", exc_info=True)
        logger.error(f"Request data: repo={getattr(request, 'repo', 'N/A')}")
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

@app.post("/compare/batch", response_model=BatchComparisonResponse)
async def batch_compare_repositories(
    request: BatchComparisonRequest,
    handler: ComparisonHandler = Depends(get_comparison_handler)
) -> BatchComparisonResponse:
    """
    Confidence-based batch repository comparison with uncertainty filtering.
    
    Processes multiple repository pairs using a two-model approach:
    1. Query llama-4-maverick first for each pair
    2. If uncertainty > 0.00000034, query gpt-4o 
    3. If gpt-4o uncertainty > 0.00077255, filter out the pair
    4. Return only results that pass uncertainty thresholds
    
    Args:
        request: Batch comparison request with pairs, parent, and optional parameters
        
    Returns:
        BatchComparisonResponse with successful and filtered comparisons
    """
    try:
        logger.info(f"Processing batch comparison: {len(request.pairs)} pairs -> {request.parent}")
        
        # Call the batch comparison handler
        response_data = await handler.handle_batch_comparison(
            pairs=request.pairs,
            parent=request.parent,
            parameters=request.parameters
        )
        
        # Create response object
        response = BatchComparisonResponse(**response_data)
        
        logger.info(f"Batch comparison complete: {response.total_successful} successful, {response.total_filtered} filtered")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in batch comparison: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing batch comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process batch comparison request")

@app.get("/special-case/status")
async def get_special_case_status(
    handler: ComparisonHandler = Depends(get_comparison_handler)
):
    """
    Get status and statistics about special case comparison functionality.
    
    Returns:
        Dictionary with special case status, validation results, and available repositories
    """
    try:
        stats = handler.get_special_case_stats()
        validation = handler.validate_special_case_data()
        
        return {
            "special_case": stats,
            "validation": validation,
            "description": {
                "enabled": "Special case uses criteria assessment data for L1 comparisons",
                "fallback": "Automatically falls back to LLM when assessment data unavailable",
                "data_source": "data/processed/criteria_assessment/detailed_assessments.json"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting special case status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve special case status")

@app.post("/special-case/toggle")
async def toggle_special_case(
    enabled: bool,
    handler: ComparisonHandler = Depends(get_comparison_handler)
):
    """
    Enable or disable special case handling.
    
    Args:
        enabled: Whether to enable special case handling
        
    Returns:
        Updated status information
    """
    try:
        handler.enable_special_case(enabled)
        
        return {
            "message": f"Special case handling {'enabled' if enabled else 'disabled'}",
            "enabled": enabled,
            "timestamp": "2024-01-15T10:30:00Z"  # Could be dynamic
        }
        
    except Exception as e:
        logger.error(f"Error toggling special case: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle special case handling")

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

@app.get("/cached-comparisons/bulk")
async def get_bulk_cached_comparisons(
    handler: ComparisonHandler = Depends(get_comparison_handler)
):
    """
    Return bulk cached comparison data for Level 1 repositories.
    Uses criteria assessment data from cached file.
    
    Returns:
        Dictionary with cached comparison results for all available repository pairs
    """
    try:
        logger.info("Processing bulk cached comparisons request")
        
        bulk_results = handler.get_bulk_cached_comparisons()
        
        logger.info(f"Bulk cached comparisons complete: {len(bulk_results.get('comparisons', []))} pairs")
        return bulk_results
        
    except Exception as e:
        logger.error(f"Error processing bulk cached comparisons: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process bulk cached comparisons")

@app.get("/cached-originality/bulk")
async def get_bulk_cached_originality(
    handler: OriginalityHandler = Depends(get_originality_handler)
):
    """
    Return bulk cached originality assessment data for all processed repositories.
    Uses pre-computed originality assessment data from individual repository files.
    
    Returns:
        Dictionary with cached originality results for all available repositories
    """
    try:
        logger.info("Processing bulk cached originality request")
        
        bulk_results = handler.get_bulk_cached_originality()
        
        logger.info(f"Bulk cached originality complete: {len(bulk_results.get('assessments', []))} repositories")
        return bulk_results
        
    except Exception as e:
        logger.error(f"Error processing bulk cached originality: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process bulk cached originality")

@app.get("/debug/comparison-methods")
async def get_comparison_methods():
    """
    Debug endpoint to show available comparison methods and their conditions.
    
    Returns:
        Information about when each comparison method is used
    """
    return {
        "comparison_methods": {
            "special_case_criteria": {
                "description": "Uses pre-computed criteria assessment scores",
                "conditions": [
                    "parent == 'ethereum'",
                    "Both repositories have assessment data available",
                    "Special case handling is enabled"
                ],
                "advantages": [
                    "Fast response (no LLM call)",
                    "Consistent scoring",
                    "Detailed reasoning based on criteria"
                ],
                "data_source": "data/processed/criteria_assessment/detailed_assessments.json"
            },
            "llm_based": {
                "description": "Uses Language Model for comparison",
                "conditions": [
                    "Special case conditions not met",
                    "Fallback when assessment data unavailable"
                ],
                "advantages": [
                    "Flexible reasoning",
                    "Can handle any repository pair",
                    "Dynamic analysis"
                ],
                "models_available": [
                    "openai/gpt-4o",
                    "meta-llama/llama-4-maverick", 
                    "x-ai/grok-3-beta",
                    "deepseek/deepseek-chat-v3-0324"
                ]
            }
        },
        "level_routing": {
            "level_1": {
                "condition": "parent == 'ethereum'",
                "description": "Ethereum ecosystem repository comparisons",
                "methods": ["special_case_criteria", "llm_based"]
            },
            "level_3": {
                "condition": "parent != 'ethereum'",
                "description": "Dependency comparisons within parent repositories",
                "methods": ["llm_based"]
            }
        }
    }

if __name__ == "__main__":
    # For development only
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )