from fastapi import APIRouter, File, UploadFile, HTTPException
import uuid
import os
from pathlib import Path
import logging

from app.core.config import settings
from app.models.schemas import UploadResponse, FileType
from app.utils.file_utils import detect_file_type, validate_file

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for data cleaning"""
    
    try:
        # Validate file
        if not validate_file(file):
            raise HTTPException(status_code=400, detail="Invalid file type or size")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Detect file type
        file_type = detect_file_type(file.filename)
        
        # Create file path
        file_extension = Path(file.filename).suffix
        file_path = settings.UPLOAD_DIR / f"{file_id}{file_extension}"
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # For unstructured files, add a note about processing requirements
        processing_note = ""
        if file_type in [FileType.PDF, FileType.IMAGE, FileType.TEXT]:
            processing_note = f"Unstructured {file_type.value} file - will be processed during profiling"
        
        logger.info(f"File uploaded: {file.filename} -> {file_id} (type: {file_type})")
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            message=f"File uploaded successfully. {processing_note}".strip()
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/files")
async def list_files():
    """List all uploaded files"""
    try:
        files = []
        for file_path in settings.UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
        return {"files": files}
    except Exception as e:
        logger.error(f"List files error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
