from fastapi import APIRouter, HTTPException
import logging
from pathlib import Path

from app.core.config import settings
from app.models.schemas import DataProfile
from app.services.profiling import ProfileService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/profile/{file_id}", response_model=DataProfile)
async def get_data_profile(file_id: str):
    """Generate data profile for uploaded file"""
    
    try:
        # Find the file
        file_path = None
        for f in settings.UPLOAD_DIR.glob(f"{file_id}.*"):
            file_path = f
            break
            
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Generate profile
        profile_service = ProfileService()
        profile = await profile_service.generate_profile(file_id, file_path)
        
        logger.info(f"Profile generated for file: {file_id}")
        return profile
        
    except Exception as e:
        logger.error(f"Profile generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Profile generation failed: {str(e)}")

@router.get("/profile/{file_id}/summary")
async def get_profile_summary(file_id: str):
    """Get a summary of the data profile"""
    
    try:
        profile = await get_data_profile(file_id)
        
        # Create summary
        summary = {
            "file_id": profile.file_id,
            "shape": f"{profile.total_rows} rows × {profile.total_columns} columns",
            "missing_percentage": sum(profile.missing_values.values()) / (profile.total_rows * profile.total_columns) * 100,
            "duplicate_percentage": profile.duplicates / profile.total_rows * 100 if profile.total_rows > 0 else 0,
            "total_outliers": sum(profile.outliers.values()),
            "data_types": profile.data_types
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Profile summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
