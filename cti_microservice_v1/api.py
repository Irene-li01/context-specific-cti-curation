import time
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import ThreatIntelRAG
from threats_endpoints import router as threats_router

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_API_Gateway")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_models["rag_engine"] = ThreatIntelRAG()
        logger.info("RAG Engine loaded.")
    except RuntimeError as e:
        logger.warning("RAG Engine disabled: %s", e)
    except Exception as e:
        logger.error("RAG Engine failed to load: %s", e)
    yield
    logger.info("Shutting down...")
    ml_models.clear()


app = FastAPI(
    title="SOC RAG Threat Intelligence API",
    description="Microservice for NL querying of curated STIX 2.1 CTI data",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)

class SourceDocument(BaseModel):
    content: str
    label: str = ""
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    status: str
    processing_time_seconds: float
    answer: str
    sources: List[SourceDocument]


def get_rag_engine() -> ThreatIntelRAG:
    engine = ml_models.get("rag_engine")
    if not engine:
        raise HTTPException(status_code=503, detail="RAG Engine unavailable or still initializing.")
    return engine


app.include_router(threats_router)


@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """Verify API and AI model status."""
    return {
        "status": "operational",
        "service": "CTI RAG Microservice",
        "rag_available": "rag_engine" in ml_models,
    }


@app.post("/api/v1/threats/query", response_model=QueryResponse, tags=["Threat Intelligence"])
async def query_threat_intel(
    request: QueryRequest,
    rag_engine: ThreatIntelRAG = Depends(get_rag_engine)
):
    """Run a natural language query against the ChromaDB vector store."""
    start_time = time.time()
    logger.info("Received query: '%s'", request.query)

    try:
        result = rag_engine.query(request.query)
        processing_time = round(time.time() - start_time, 2)
        logger.info("Query processed in %.2fs", processing_time)

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["answer"])

        return QueryResponse(
            status="success",
            processing_time_seconds=processing_time,
            answer=result["answer"],
            sources=[SourceDocument(**src) for src in result["sources"]]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled exception during query: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing threat intelligence query.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
