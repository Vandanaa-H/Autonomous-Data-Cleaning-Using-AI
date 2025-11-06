from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uuid
from pathlib import Path
import logging

from app.core.config import settings
from app.api import upload, profile, clean, download, report
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Autonomous Data Cleaning API",
    description="AI-powered data cleaning system with automatic quality detection and repair",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(profile.router, prefix="/api/v1", tags=["profile"])
app.include_router(clean.router, prefix="/api/v1", tags=["clean"])
app.include_router(download.router, prefix="/api/v1", tags=["download"])
app.include_router(report.router, prefix="/api/v1", tags=["report"])

# Also include routers without prefix for direct access
app.include_router(upload.router, tags=["upload-direct"])
app.include_router(profile.router, tags=["profile-direct"])
app.include_router(clean.router, tags=["clean-direct"])
app.include_router(download.router, tags=["download-direct"])
app.include_router(report.router, tags=["report-direct"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Autonomous Data Cleaning API",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "upload_dir": str(settings.UPLOAD_DIR),
        "output_dir": str(settings.OUTPUT_DIR)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
