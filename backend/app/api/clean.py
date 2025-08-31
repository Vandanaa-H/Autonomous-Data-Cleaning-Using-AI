from fastapi import APIRouter, HTTPException
import logging
from pathlib import Path

from app.core.config import settings
from app.models.schemas import CleaningRequest, CleaningReport
from app.services.cleaning_engine import CleaningEngine

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/clean/{file_id}", response_model=CleaningReport)
async def clean_dataset(file_id: str, request: CleaningRequest = None):
    """Clean the uploaded dataset"""
    
    try:
        # Find the file
        file_path = None
        for f in settings.UPLOAD_DIR.glob(f"{file_id}.*"):
            file_path = f
            break
            
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Initialize cleaning engine
        cleaning_engine = CleaningEngine()
        
        # Perform cleaning
        report = await cleaning_engine.clean_dataset(
            file_id=file_id,
            file_path=file_path,
            target_task=request.target_task if request else None,
            custom_rules=request.custom_rules if request else None
        )
        
        logger.info(f"Dataset cleaned successfully: {file_id}")
        return report
        
    except Exception as e:
        logger.error(f"Cleaning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {str(e)}")

@router.get("/clean/{file_id}/status")
async def get_cleaning_status(file_id: str):
    """Get the status of cleaning process"""
    
    try:
        # Check if cleaned file exists
        cleaned_files = list(settings.OUTPUT_DIR.glob(f"{file_id}_cleaned.*"))
        
        if cleaned_files:
            return {
                "file_id": file_id,
                "status": "completed",
                "cleaned_file": cleaned_files[0].name
            }
        else:
            return {
                "file_id": file_id,
                "status": "not_started",
                "cleaned_file": None
            }
            
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
