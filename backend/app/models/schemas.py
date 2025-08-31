from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class FileType(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"

class FileStatus(str, Enum):
    UPLOADED = "uploaded"
    PROFILED = "profiled"
    CLEANED = "cleaned"
    ERROR = "error"

class FileInfo(BaseModel):
    file_id: str
    filename: str
    file_type: FileType
    file_size: int
    upload_time: datetime
    status: FileStatus
    
class DataProfile(BaseModel):
    file_id: str
    total_rows: int
    total_columns: int
    column_info: Dict[str, Any]
    missing_values: Dict[str, int]
    duplicates: int
    outliers: Dict[str, int]
    data_types: Dict[str, str]
    
class DetectedIssue(BaseModel):
    issue_type: str
    column: Optional[str] = None
    description: str
    severity: str  # "low", "medium", "high"
    affected_rows: int
    
class CleaningStrategy(BaseModel):
    strategy_name: str
    description: str
    parameters: Dict[str, Any] = {}
    
class CleaningAction(BaseModel):
    issue: DetectedIssue
    strategy: CleaningStrategy
    result: Dict[str, Any]
    
class CleaningReport(BaseModel):
    file_id: str
    original_file: str
    cleaned_file: str
    processing_time: float
    issues_detected: List[DetectedIssue]
    actions_taken: List[CleaningAction]
    before_stats: Dict[str, Any]
    after_stats: Dict[str, Any]
    
class UploadResponse(BaseModel):
    file_id: str
    filename: str
    message: str
    
class CleaningRequest(BaseModel):
    file_id: str
    target_task: Optional[str] = None  # "classification", "regression", etc.
    custom_rules: Optional[Dict[str, Any]] = None
