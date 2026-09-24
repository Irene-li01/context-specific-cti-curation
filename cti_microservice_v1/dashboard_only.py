"""Lightweight entrypoint that boots ONLY the threats / dashboard router.

Use this when iterating on the frontend or stats endpoints — it skips
the heavy RAG engine startup (ChromaDB + transformer model load) so the
server is ready in milliseconds instead of minutes.

Run::

    cd cti_microservice_v1
    python dashboard_only.py            # serves on http://0.0.0.0:8000/

For production, use ``api.py`` (which mounts both the dashboard router
and the RAG query endpoint).
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from threats_endpoints import router as threats_router


# --------------------------------------------------------------------------- #
# Environment-driven configuration. All values have sensible dev defaults so
# ``python dashboard_only.py`` works on a fresh clone with no setup; production
# deployments override via systemd / docker env vars.
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.getenv("API_LOG_LEVEL", "INFO").upper()
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Comma-separated list. Use a single ``*`` (default) only for local dev.
DASHBOARD_ORIGINS = [
    origin.strip()
    for origin in os.getenv("DASHBOARD_ORIGINS", "*").split(",")
    if origin.strip()
]
ALLOW_CREDENTIALS = os.getenv("DASHBOARD_ALLOW_CREDENTIALS", "true").lower() == "true"

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Dashboard_API")

app = FastAPI(
    title="CCTI Dashboard (lightweight mode)",
    description="Serves the noise-filtered CTI dataset and the static dashboard. "
                "The RAG / NLP query endpoint is NOT loaded in this mode.",
    version="1.0.0-dashboard",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DASHBOARD_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_csp_header(request: Request, call_next):
    response = await call_next(request)
    # Allow unsafe-eval for Plotly/Chart.js and CDN scripts
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
        "https://cdn.jsdelivr.net https://cdn.plot.ly https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self' https://cdn.jsdelivr.net https://cdn.plot.ly https://unpkg.com;"
    )
    return response

app.include_router(threats_router)


@app.get("/api/v1/health", tags=["System"])
async def health_check():
    return {"status": "operational", "service": "CCTI Dashboard", "rag_loaded": False}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting dashboard on %s:%s (CORS origins: %s)",
                API_HOST, API_PORT, DASHBOARD_ORIGINS)
    uvicorn.run("dashboard_only:app", host=API_HOST, port=API_PORT, reload=False)
