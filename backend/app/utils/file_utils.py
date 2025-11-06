from fastapi import UploadFile
from pathlib import Path
import os
from typing import Optional

from app.models.schemas import FileType

# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    '.csv': FileType.CSV,
    '.xlsx': FileType.EXCEL,
    '.xls': FileType.EXCEL,
    '.json': FileType.JSON,
    '.txt': FileType.TEXT,
    '.pdf': FileType.PDF,
    '.png': FileType.IMAGE,
    '.jpg': FileType.IMAGE,
    '.jpeg': FileType.IMAGE,
    '.tiff': FileType.IMAGE,
    '.bmp': FileType.IMAGE
}


def detect_file_type(filename: str) -> FileType:
    """Detect file type from filename extension"""

    if not filename:
        raise ValueError("Filename cannot be empty")

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    return ALLOWED_EXTENSIONS[extension]


def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""

    try:
        # Check filename
        if not file.filename:
            return False

        # Check file extension
        extension = Path(file.filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return False

        # Check file size (if available)
        if hasattr(file, 'size') and file.size:
            if file.size > MAX_FILE_SIZE:
                return False

        return True

    except Exception:
        return False


def get_file_info(file_path: Path) -> dict:
    """Get detailed file information"""

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stat = file_path.stat()

    return {
        "name": file_path.name,
        "size": stat.st_size,
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "extension": file_path.suffix.lower(),
        "type": detect_file_type(file_path.name)
    }


def is_text_file(file_path: Path) -> bool:
    """Check if file is a text file"""

    text_extensions = {'.csv', '.txt', '.json'}
    return file_path.suffix.lower() in text_extensions


def is_binary_file(file_path: Path) -> bool:
    """Check if file is a binary file"""

    binary_extensions = {'.xlsx', '.xls', '.pdf',
                         '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    return file_path.suffix.lower() in binary_extensions
