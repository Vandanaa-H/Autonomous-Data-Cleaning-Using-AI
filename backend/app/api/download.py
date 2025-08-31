from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import logging
from pathlib import Path

from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/download/{file_id}")
async def download_cleaned_file(file_id: str):
    """Download the cleaned dataset"""
    
    try:
        # Find cleaned file
        cleaned_files = list(settings.OUTPUT_DIR.glob(f"{file_id}_cleaned.*"))
        
        if not cleaned_files:
            raise HTTPException(status_code=404, detail="Cleaned file not found")
        
        file_path = cleaned_files[0]
        
        # Return file
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.get("/download/{file_id}/original")
async def download_original_file(file_id: str):
    """Download the original uploaded file"""
    
    try:
        # Find original file
        file_path = None
        for f in settings.UPLOAD_DIR.glob(f"{file_id}.*"):
            file_path = f
            break
            
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Original file not found")
        
        # Return file
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Download original error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
