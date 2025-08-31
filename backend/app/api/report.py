from fastapi import APIRouter, HTTPException
import logging
from pathlib import Path
import json

from app.core.config import settings
from app.models.schemas import CleaningReport

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/report/{file_id}", response_model=CleaningReport)
async def get_cleaning_report(file_id: str):
    """Get the detailed cleaning report"""
    
    try:
        # Find report file
        report_path = settings.OUTPUT_DIR / f"{file_id}_report.json"
        
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Load report
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        # Convert to CleaningReport model
        report = CleaningReport(**report_data)
        
        return report
        
    except Exception as e:
        logger.error(f"Report retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report retrieval failed: {str(e)}")

@router.get("/report/{file_id}/html")
async def get_report_html(file_id: str):
    """Get the HTML version of the cleaning report"""
    
    try:
        # Find HTML report
        html_path = settings.OUTPUT_DIR / f"{file_id}_report.html"
        
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="HTML report not found")
        
        # Read HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return {"html_content": html_content}
        
    except Exception as e:
        logger.error(f"HTML report error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/{file_id}/summary")
async def get_report_summary(file_id: str):
    """Get a summary of the cleaning report"""
    
    try:
        report = await get_cleaning_report(file_id)
        
        summary = {
            "file_id": report.file_id,
            "processing_time": report.processing_time,
            "issues_found": len(report.issues_detected),
            "actions_taken": len(report.actions_taken),
            "improvement_metrics": {
                "missing_values_fixed": report.before_stats.get("missing_values", 0) - report.after_stats.get("missing_values", 0),
                "duplicates_removed": report.before_stats.get("duplicates", 0) - report.after_stats.get("duplicates", 0),
                "outliers_handled": report.before_stats.get("outliers", 0) - report.after_stats.get("outliers", 0)
            }
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Report summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
