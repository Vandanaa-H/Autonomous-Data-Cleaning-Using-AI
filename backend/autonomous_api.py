"""
Professional AI-Powered Data Cleaning Backend
A fully autonomous data cleaning system with database persistence,
cloud storage, and enhanced security logging
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import numpy as np
import os
import uuid
import shutil
from pathlib import Path
import io
import json
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import hashlib
import time

# Import our custom modules
from database import DatabaseManager, get_db, init_database, SQLALCHEMY_AVAILABLE
from cloud_storage import storage_manager
from secure_logging import secure_logger, request_logger, log_function_call, log_audit_trail

# Import unstructured data processor
try:
    from app.services.nlp.unstructured_processor import UnstructuredDataProcessor
    from app.models.schemas import FileType
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    secure_logger.warning("Unstructured data processor not available")

# Configure logging
secure_logger.info("Starting AI Data Cleaning Backend with enhanced features")

# Create FastAPI app
app = FastAPI(
    title="Autonomous AI Data Cleaning API",
    description="Fully autonomous data cleaning with database persistence and cloud storage",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize database on startup


@app.on_event("startup")
async def startup_event():
    """Initialize database and storage on startup"""
    # Database is optional - app will work with in-memory storage
    secure_logger.warning("Running without database - using in-memory storage")
    # if SQLALCHEMY_AVAILABLE:
    #     success = init_database()
    #     if success:
    #         secure_logger.info("Database initialized successfully")
    #     else:
    #         secure_logger.error("Database initialization failed")
    # else:
    #     secure_logger.warning("Database not available, using fallback storage")

# Middleware for request logging


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    response = await call_next(request)

    process_time = time.time() - start_time

    request_logger.log_request(
        method=request.method,
        path=str(request.url.path),
        status_code=response.status_code,
        response_time=process_time,
        user_agent=user_agent,
        ip_address=client_ip,
        request_id=request_id
    )

    return response

# Database dependency with fallback


def get_database():
    """Get database session with fallback to None"""
    return None  # Always return None to use in-memory storage


# In-memory storage for fallback (when database is not available)
file_storage = {}


class AutonomousDataCleaner:
    """AI-powered autonomous data cleaner"""

    def __init__(self):
        self.cleaning_strategies = {
            'missing_values': ['drop_rows', 'mean_imputation', 'median_imputation', 'mode_imputation', 'knn_imputation'],
            'duplicates': ['drop_duplicates', 'keep_first', 'keep_last'],
            'outliers': ['remove_outliers', 'cap_outliers', 'log_transform'],
            'text_issues': ['normalize_case', 'strip_whitespace', 'spell_correction']
        }

    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive autonomous data analysis with detailed issue detection"""

        # Initialize analysis structure
        analysis = {
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_by_column': {},
            'issues': [],
            'quality_score': 100.0,
            'summary': {}
        }

        total_cells = len(df) * len(df.columns)
        issues_found = []
        quality_deductions = 0

        # 1. MISSING VALUES DETECTION - ENHANCED
        missing_analysis = {}
        for col in df.columns:
            # Check for NaN, empty strings, and whitespace-only strings
            missing_mask = df[col].isnull() | (df[col].astype(str).str.strip() == '') | (
                df[col].astype(str).str.strip() == 'nan')
            missing_count = missing_mask.sum()

            if missing_count > 0:
                missing_analysis[col] = int(missing_count)
                missing_pct = (missing_count / len(df)) * 100

                # Add detailed issue
                issues_found.append({
                    'type': 'missing_values',
                    'column': col,
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2),
                    'description': f"Column '{col}' has {missing_count} missing values ({missing_pct:.1f}%)",
                    'severity': 'high' if missing_pct > 30 else 'medium' if missing_pct > 10 else 'low'
                })

                # Quality deduction
                # Max 25 points per column
                quality_deductions += min(missing_pct * 0.5, 25)
                quality_deductions += min(missing_pct * 0.5, 15)

        analysis['missing_by_column'] = missing_analysis

        # 2. DUPLICATE ROWS DETECTION
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            dup_pct = (duplicate_count / len(df)) * 100
            issues_found.append({
                'type': 'duplicates',
                'column': 'all_columns',
                'count': int(duplicate_count),
                'percentage': round(dup_pct, 2),
                'description': f"Found {duplicate_count} duplicate rows ({dup_pct:.1f}% of data)",
                'severity': 'high' if dup_pct > 10 else 'medium' if dup_pct > 5 else 'low'
            })
            quality_deductions += min(dup_pct * 0.8, 20)

        # 3. OUTLIERS DETECTION (for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 3:  # Need at least 4 values for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                if IQR > 0:  # Avoid division by zero
                    outlier_mask = (df[col] < Q1 - 1.5 *
                                    IQR) | (df[col] > Q3 + 1.5 * IQR)
                    outlier_count = outlier_mask.sum()

                    if outlier_count > 0:
                        outlier_pct = (outlier_count /
                                       len(df[col].dropna())) * 100
                        issues_found.append({
                            'type': 'outliers',
                            'column': col,
                            'count': int(outlier_count),
                            'percentage': round(outlier_pct, 2),
                            'description': f"Column '{col}' has {outlier_count} statistical outliers ({outlier_pct:.1f}%)",
                            'severity': 'medium' if outlier_pct > 5 else 'low'
                        })
                        quality_deductions += min(outlier_pct * 0.3, 10)

        # 4. TEXT QUALITY ISSUES
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            col_data = df[col].dropna().astype(str)
            if len(col_data) > 0:

                # Check for whitespace issues
                whitespace_issues = (col_data.str.startswith(
                    ' ') | col_data.str.endswith(' ')).sum()
                if whitespace_issues > 0:
                    whitespace_pct = (whitespace_issues / len(col_data)) * 100
                    issues_found.append({
                        'type': 'whitespace',
                        'column': col,
                        'count': int(whitespace_issues),
                        'percentage': round(whitespace_pct, 2),
                        'description': f"Column '{col}' has {whitespace_issues} values with leading/trailing whitespace",
                        'severity': 'low'
                    })
                    quality_deductions += min(whitespace_pct * 0.2, 5)

                # Check for case inconsistencies
                unique_values = col_data.unique()
                unique_lower = col_data.str.lower().unique()
                if len(unique_values) > len(unique_lower):
                    case_issues = len(unique_values) - len(unique_lower)
                    issues_found.append({
                        'type': 'case_inconsistency',
                        'column': col,
                        'count': case_issues,
                        'percentage': round((case_issues / len(unique_values)) * 100, 2),
                        'description': f"Column '{col}' has inconsistent capitalization ({case_issues} case variations)",
                        'severity': 'low'
                    })
                    quality_deductions += min(case_issues * 0.5, 8)

                # Check for email format issues
                if 'email' in col.lower():
                    # Basic email pattern validation
                    email_pattern = r'^[^@]+@[^@]+\.[^@]+$'
                    valid_emails = col_data.str.contains(
                        email_pattern, na=False, regex=True)
                    invalid_emails = (~valid_emails).sum()
                    if invalid_emails > 0:
                        invalid_pct = (invalid_emails / len(col_data)) * 100
                        issues_found.append({
                            'type': 'invalid_email_format',
                            'column': col,
                            'count': int(invalid_emails),
                            'percentage': round(invalid_pct, 2),
                            'description': f"Column '{col}' has {invalid_emails} invalid email formats",
                            'severity': 'high'
                        })
                        quality_deductions += min(invalid_pct * 0.8, 15)

                # Check for phone format issues
                if 'phone' in col.lower():
                    # Check for consistent phone formatting
                    phone_patterns = [
                        # Standard US formats
                        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
                        # (xxx) xxx-xxxx format
                        r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}'
                    ]
                    valid_phones = col_data.str.contains(
                        '|'.join(phone_patterns), na=False, regex=True)
                    invalid_phones = (~valid_phones).sum()
                    if invalid_phones > 0:
                        invalid_pct = (invalid_phones / len(col_data)) * 100
                        issues_found.append({
                            'type': 'invalid_phone_format',
                            'column': col,
                            'count': int(invalid_phones),
                            'percentage': round(invalid_pct, 2),
                            'description': f"Column '{col}' has {invalid_phones} invalid phone number formats",
                            'severity': 'medium'
                        })
                        quality_deductions += min(invalid_pct * 0.6, 12)

                # Check for empty strings
                empty_strings = (col_data == '').sum()
                if empty_strings > 0:
                    empty_pct = (empty_strings / len(col_data)) * 100
                    issues_found.append({
                        'type': 'empty_strings',
                        'column': col,
                        'count': int(empty_strings),
                        'percentage': round(empty_pct, 2),
                        'description': f"Column '{col}' has {empty_strings} empty string values",
                        'severity': 'medium'
                    })
                    quality_deductions += min(empty_pct * 0.4, 10)

        # 5. DATA TYPE INCONSISTENCIES
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data is stored as text
                non_null_data = df[col].dropna().astype(str)
                if len(non_null_data) > 0:
                    # Try to convert to numeric
                    try:
                        numeric_convertible = pd.to_numeric(
                            non_null_data, errors='coerce').notna().sum()
                        # 80% can be converted
                        if numeric_convertible > len(non_null_data) * 0.8:
                            issues_found.append({
                                'type': 'data_type_mismatch',
                                'column': col,
                                'count': int(numeric_convertible),
                                'percentage': round((numeric_convertible / len(non_null_data)) * 100, 2),
                                'description': f"Column '{col}' contains numeric data stored as text",
                                'severity': 'medium'
                            })
                            quality_deductions += 5
                    except:
                        pass

        # Calculate final quality score
        final_quality = max(0, 100 - quality_deductions)
        analysis['quality_score'] = round(final_quality, 1)
        analysis['issues'] = issues_found

        # Summary statistics
        analysis['summary'] = {
            'total_issues': len(issues_found),
            'high_severity': len([i for i in issues_found if i.get('severity') == 'high']),
            'medium_severity': len([i for i in issues_found if i.get('severity') == 'medium']),
            'low_severity': len([i for i in issues_found if i.get('severity') == 'low']),
            'quality_deductions': round(quality_deductions, 2)
        }

        secure_logger.info(
            f"Analysis complete: {len(issues_found)} issues found, quality score: {final_quality:.1f}%")

        return analysis

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """AI generates autonomous cleaning recommendations"""
        recommendations = []

        # Missing values recommendations
        for col, missing_count in analysis['missing_values'].items():
            missing_pct = (missing_count / analysis['shape']['rows']) * 100
            if missing_pct > 70:
                recommendations.append(
                    f" Remove column '{col}' (70%+ missing)")
            elif missing_pct > 30:
                recommendations.append(
                    f" Advanced imputation for '{col}' (KNN/Iterative)")
            else:
                recommendations.append(
                    f" Smart imputation for '{col}' (mean/median/mode)")

        # Duplicate recommendations
        if analysis['duplicates'] > 0:
            recommendations.append(
                f" Remove {analysis['duplicates']} duplicate rows")

        # Outlier recommendations
        for col, outlier_count in analysis['outliers'].items():
            outlier_pct = (outlier_count / analysis['shape']['rows']) * 100
            if outlier_pct > 10:
                recommendations.append(
                    f" Cap outliers in '{col}' (preserve data)")
            else:
                recommendations.append(
                    f" Remove outliers in '{col}' (likely errors)")

        # Text recommendations
        for col, issues in analysis['text_issues'].items():
            if 'case_inconsistency' in issues:
                recommendations.append(f" Normalize case in '{col}'")
            if 'whitespace_issues' in issues:
                recommendations.append(f" Trim whitespace in '{col}'")

        return recommendations

    def _detect_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Detect specific data quality issues"""
        issues = []

        if analysis['missing_values']:
            issues.append("Missing Values Detected")
        if analysis['duplicates'] > 0:
            issues.append("Duplicate Rows Found")
        if analysis['outliers']:
            issues.append("Statistical Outliers Present")
        if analysis['text_issues']:
            issues.append("Text Quality Issues")

        if not issues:
            issues.append("No Major Issues Detected")

        return issues

    def clean_data_autonomously(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Fully autonomous data cleaning with AI decision making"""

        original_shape = df.shape
        cleaning_report = {
            'original_shape': original_shape,
            'steps_applied': [],
            'improvements': {},
            'final_quality_score': 0
        }

        # Step 1: Handle missing values autonomously
        df, missing_report = self._handle_missing_values_ai(df)
        if missing_report['changes_made']:
            cleaning_report['steps_applied'].append(missing_report)

        # Step 2: Remove duplicates autonomously
        df, duplicate_report = self._handle_duplicates_ai(df)
        if duplicate_report['changes_made']:
            cleaning_report['steps_applied'].append(duplicate_report)

        # Step 3: Handle outliers autonomously
        df, outlier_report = self._handle_outliers_ai(df)
        if outlier_report['changes_made']:
            cleaning_report['steps_applied'].append(outlier_report)

        # Step 4: Fix text issues autonomously
        df, text_report = self._handle_text_issues_ai(df)
        if text_report['changes_made']:
            cleaning_report['steps_applied'].append(text_report)

        # Final analysis
        final_analysis = self.analyze_data(df)
        cleaning_report['final_shape'] = df.shape
        cleaning_report['final_quality_score'] = final_analysis['quality_score']
        cleaning_report['improvements'] = {
            'rows_removed': original_shape[0] - df.shape[0],
            # Assume baseline of 50
            'quality_improvement': final_analysis['quality_score'] - 50,
            'issues_resolved': len(cleaning_report['steps_applied'])
        }

        return df, cleaning_report

    def _handle_missing_values_ai(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """AI-powered missing value handling - FAST & EFFICIENT"""
        report = {'step': 'Missing Values',
                  'changes_made': False, 'details': []}

        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100

            if missing_pct > 70:
                # Drop columns with >70% missing
                df = df.drop(columns=[col])
                report['details'].append(
                    f"Dropped column '{col}' (70%+ missing)")
                report['changes_made'] = True

            elif missing_pct > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric: use median for robustness
                    df[col] = df[col].fillna(df[col].median())
                    report['details'].append(f"Filled '{col}' with median")
                    report['changes_made'] = True
                else:
                    # Categorical: use efficient group-wise mode imputation
                    try:
                        # Find best parent column for group-wise imputation
                        cat_cols = df.select_dtypes(
                            include=['object']).columns.tolist()
                        best_parent = None

                        # Look for common parent columns (Category, Department, etc.)
                        for potential_parent in cat_cols:
                            if potential_parent == col:
                                continue
                            # Check if this parent has reasonable cardinality and isn't too sparse
                            if df[potential_parent].nunique() <= 30 and df[potential_parent].notna().sum() > len(df) * 0.5:
                                best_parent = potential_parent
                                break

                        filled_count = 0
                        if best_parent:
                            # Group-wise mode imputation
                            missing_mask = df[col].isnull()
                            for parent_val in df[best_parent].unique():
                                if pd.notna(parent_val):
                                    group_mask = (
                                        df[best_parent] == parent_val) & missing_mask
                                    if group_mask.any():
                                        # Get mode for this group
                                        group_mode = df.loc[(
                                            df[best_parent] == parent_val) & df[col].notna(), col].mode()
                                        if not group_mode.empty:
                                            df.loc[group_mask,
                                                   col] = group_mode.iloc[0]
                                            filled_count += group_mask.sum()

                        # Fill any remaining with global mode
                        remaining_missing = df[col].isnull().sum()
                        if remaining_missing > 0:
                            global_mode = df[col].mode()
                            if not global_mode.empty:
                                df[col] = df[col].fillna(global_mode.iloc[0])
                                filled_count += remaining_missing

                        if filled_count > 0:
                            strategy = f"group-wise mode (parent: {best_parent})" if best_parent else "global mode"
                            report['details'].append(
                                f"Filled '{col}' with {strategy} ({filled_count} values)")
                            report['changes_made'] = True

                    except Exception as e:
                        # Fallback to simple mode
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col] = df[col].fillna(mode_val.iloc[0])
                            report['details'].append(
                                f"Filled '{col}' with global mode (fallback)")
                            report['changes_made'] = True

        return df, report

    def _handle_duplicates_ai(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """AI-powered duplicate removal"""
        original_len = len(df)
        df = df.drop_duplicates(keep='first')
        duplicates_removed = original_len - len(df)

        report = {
            'step': 'Duplicates',
            'changes_made': duplicates_removed > 0,
            'details': [f"Removed {duplicates_removed} duplicate rows"] if duplicates_removed > 0 else []
        }

        return df, report

    def _handle_outliers_ai(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """AI-powered outlier handling"""
        report = {'step': 'Outliers', 'changes_made': False, 'details': []}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if len(df[col].dropna()) > 10:  # Need sufficient data
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                outlier_count = len(
                    df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
                outlier_pct = (outlier_count / len(df)) * 100

                if outlier_pct > 10:
                    # Cap outliers (preserve data)
                    lower_cap = Q1 - 1.5 * IQR
                    upper_cap = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
                    report['details'].append(
                        f"Capped outliers in '{col}' ({outlier_count} values)")
                    report['changes_made'] = True

                elif outlier_pct > 0 and outlier_pct <= 5:
                    # Remove extreme outliers
                    df = df[(df[col] >= Q1 - 1.5 * IQR) &
                            (df[col] <= Q3 + 1.5 * IQR)]
                    report['details'].append(
                        f"Removed extreme outliers in '{col}' ({outlier_count} rows)")
                    report['changes_made'] = True

        return df, report

    def _handle_text_issues_ai(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """AI-powered text cleaning"""
        report = {'step': 'Text Issues', 'changes_made': False, 'details': []}

        text_cols = df.select_dtypes(include=['object']).columns

        for col in text_cols:
            # Strip whitespace
            original_values = df[col].astype(str).copy()
            df[col] = df[col].astype(str).str.strip()

            if not df[col].equals(original_values):
                report['details'].append(f"Trimmed whitespace in '{col}'")
                report['changes_made'] = True

            # Normalize case if inconsistent
            if len(df[col].dropna()) > 0:
                unique_lower = df[col].str.lower().nunique()
                unique_original = df[col].nunique()

                if unique_lower < unique_original * 0.8:  # Significant case inconsistency
                    df[col] = df[col].str.title()
                    report['details'].append(
                        f"Normalized case in '{col}' to title case")
                    report['changes_made'] = True

        return df, report


# Initialize the cleaner
cleaner = AutonomousDataCleaner()


@app.get("/")
async def root():
    """API Health Check"""
    return {
        "service": "Autonomous AI Data Cleaning API",
        "version": "2.0.0",
        "status": "Online",
        "features": [
            "Intelligent Data Analysis",
            "Autonomous Cleaning",
            "Professional Reports",
            "2-Click Operation"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "DataClean Pro API"
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), request: Request = None, db=Depends(get_database)):
    """Upload file for autonomous cleaning with database persistence"""

    try:
        # Get client information for logging
        client_ip = request.client.host if request and request.client else "unknown"
        user_agent = request.headers.get(
            "user-agent", "unknown") if request else "unknown"

        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Read file content
        file_content = await file.read()

        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Save file using storage manager (handles both local and cloud)
        file_info = storage_manager.save_file(
            file_data=file_content,
            filename=file.filename,
            subfolder="uploads"
        )

        # Determine file type and load data
        file_extension = file.filename.lower().split('.')[-1]
        df = None

        # Handle structured data formats
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_content))
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(file_content))
        elif file_extension == 'json':
            df = pd.read_json(io.BytesIO(file_content))

        # Handle unstructured data formats
        elif file_extension in ['pdf', 'png', 'jpg', 'jpeg', 'html', 'txt']:
            if not UNSTRUCTURED_AVAILABLE:
                raise HTTPException(
                    status_code=400,
                    detail="Unstructured data processing not available. Please install required dependencies."
                )

            try:
                # Save temporary file for processing
                temp_file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
                with open(temp_file_path, 'wb') as f:
                    f.write(file_content)

                # Determine FileType
                if file_extension == 'pdf':
                    file_type = FileType.PDF
                elif file_extension in ['png', 'jpg', 'jpeg']:
                    file_type = FileType.IMAGE
                elif file_extension == 'html':
                    file_type = FileType.TEXT  # HTML processed as text
                else:  # txt
                    file_type = FileType.TEXT

                # Process unstructured file
                processor = UnstructuredDataProcessor()
                if processor.can_process(file_type):
                    df = processor.process_file(temp_file_path, file_type)
                    secure_logger.info(
                        f"Processed unstructured file: {file.filename} -> {len(df)} rows")
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot process {file_extension} files. Required libraries not installed."
                    )

                # Clean up temporary file
                if temp_file_path.exists():
                    temp_file_path.unlink()

            except RuntimeError as re:
                # Handle specific runtime errors (like Tesseract not found)
                temp_file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
                if temp_file_path.exists():
                    temp_file_path.unlink()

                # Provide user-friendly error message
                error_msg = str(re)
                if "Tesseract" in error_msg:
                    raise HTTPException(
                        status_code=400,
                        detail="Image processing requires Tesseract OCR to be installed. Please use PDF or text files, or install Tesseract from: https://github.com/tesseract-ocr/tesseract"
                    )
                else:
                    raise HTTPException(status_code=500, detail=error_msg)

            except Exception as e:
                # Clean up temporary file on error
                temp_file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
                if temp_file_path.exists():
                    temp_file_path.unlink()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing {file_extension} file: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_extension}. Supported formats: csv, xlsx, xls, json, pdf, png, jpg, jpeg, html, txt"
            )

        if df is None or df.empty:
            raise HTTPException(
                status_code=400,
                detail="No data could be extracted from the file"
            )

        # Autonomous analysis
        analysis = cleaner.analyze_data(df)

        # Always use in-memory storage for now (database operations disabled)
        file_storage[file_id] = {
            'filename': file.filename,
            'original_data': df,
            'analysis': analysis,
            'upload_time': datetime.now().isoformat(),
            'file_info': file_info
        }
        actual_file_id = file_id
        session_id = None

        # Log upload event
        request_logger.log_file_upload(
            filename=file.filename,
            file_size=file_info['size'],
            file_hash=file_hash,
            ip_address=client_ip,
            session_id=session_id
        )

        log_audit_trail(
            action="FILE_UPLOAD",
            resource=f"file:{file.filename}",
            details={
                'file_id': actual_file_id,
                'size': file_info['size'],
                'hash': file_hash,
                'cloud_stored': file_info.get('cloud_url') is not None
            }
        )

        secure_logger.info(
            f"File uploaded and analyzed: {file.filename} (ID: {actual_file_id})")

        return {
            "file_id": actual_file_id,
            "session_id": session_id,
            "filename": file.filename,
            "status": "Analyzed",
            "analysis": analysis,
            "file_size": file_info['size'],
            "cloud_stored": file_info.get('cloud_url') is not None
        }

    except Exception as e:
        secure_logger.error(
            f"Upload error for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/analyze/{file_id}")
async def get_analysis(file_id: str):
    """Get data analysis for uploaded file"""
    try:
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")

        return file_storage[file_id]['analysis']

    except Exception as e:
        secure_logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/clean/{file_id}")
async def clean_data_autonomous(file_id: str):
    """Autonomous data cleaning with AI - No user input required!"""

    try:
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")

        # Get original data
        original_df = file_storage[file_id]['original_data'].copy()
        original_analysis = file_storage[file_id]['analysis']

        # Perform autonomous cleaning
        cleaned_df, cleaning_steps = cleaner.clean_data_autonomously(
            original_df)

        # Analyze cleaned data for quality comparison
        cleaned_analysis = cleaner.analyze_data(cleaned_df)

        # Save cleaned data
        output_path = OUTPUT_DIR / f"cleaned_{file_id}.csv"
        cleaned_df.to_csv(output_path, index=False)

        # Create comprehensive cleaning report
        cleaning_report = {
            "before_quality": original_analysis,
            "after_quality": cleaned_analysis,
            "steps_applied": cleaning_steps,
            "improvement": cleaned_analysis['quality_score'] - original_analysis['quality_score'],
            "shape_change": {
                "before": {"rows": len(original_df), "columns": len(original_df.columns)},
                "after": {"rows": len(cleaned_df), "columns": len(cleaned_df.columns)}
            }
        }

        # Update storage
        file_storage[file_id]['cleaned_data'] = cleaned_df
        file_storage[file_id]['cleaning_report'] = cleaning_report
        file_storage[file_id]['output_path'] = output_path

        secure_logger.info(
            f"Autonomous cleaning completed for file: {file_id}")
        secure_logger.info(
            f"Quality improved from {original_analysis['quality_score']:.1f}% to {cleaned_analysis['quality_score']:.1f}%")

        return {
            "file_id": file_id,
            "status": "Autonomously Cleaned",
            "before_quality": original_analysis,
            "after_quality": cleaned_analysis,
            "steps_applied": cleaning_steps,
            "improvement": cleaned_analysis['quality_score'] - original_analysis['quality_score'],
            "shape_change": {
                "before": {"rows": len(original_df), "columns": len(original_df.columns)},
                "after": {"rows": len(cleaned_df), "columns": len(cleaned_df.columns)}
            },
            "download_ready": True
        }

    except Exception as e:
        secure_logger.error(f"Cleaning error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Cleaning failed: {str(e)}")


@app.get("/download/{file_id}/{data_type}")
async def download_file(file_id: str, data_type: str):
    """Download original or cleaned dataset"""

    try:
        # In-memory storage is best-effort; during reloads it can be empty.
        # We'll fall back to disk artifacts when possible.
        if file_id not in file_storage:
            if data_type == "cleaned":
                # Fallback to cleaned CSV saved during cleaning
                fallback_path = OUTPUT_DIR / f"cleaned_{file_id}.csv"
                if fallback_path.exists():
                    return FileResponse(
                        path=fallback_path,
                        filename=f"cleaned_{fallback_path.name.replace('cleaned_', '')}",
                        media_type='text/csv'
                    )
            # If no fallback, report not found
            raise HTTPException(status_code=404, detail="File not found")

        if data_type == "original":
            df = file_storage[file_id]['original_data']
            filename = f"original_{file_storage[file_id]['filename']}"
        elif data_type == "cleaned":
            if file_id in file_storage and 'cleaned_data' in file_storage[file_id]:
                df = file_storage[file_id]['cleaned_data']
                filename = f"cleaned_{file_storage[file_id]['filename']}"
                # Create temporary file and stream
                output_path = OUTPUT_DIR / f"temp_{data_type}_{file_id}.csv"
                df.to_csv(output_path, index=False)
                return FileResponse(
                    path=output_path,
                    filename=filename,
                    media_type='text/csv'
                )
            # Fallback to file saved on disk by cleaner
            fallback_path = OUTPUT_DIR / f"cleaned_{file_id}.csv"
            if fallback_path.exists():
                return FileResponse(
                    path=fallback_path,
                    filename=f"cleaned_{fallback_path.name.replace('cleaned_', '')}",
                    media_type='text/csv'
                )
            raise HTTPException(
                status_code=404, detail="Cleaned data not available")
        else:
            raise HTTPException(
                status_code=400, detail="Invalid data type. Use 'original' or 'cleaned'")

        # Default in-memory flow for 'original'
        output_path = OUTPUT_DIR / f"temp_{data_type}_{file_id}.csv"
        df.to_csv(output_path, index=False)
        return FileResponse(path=output_path, filename=filename, media_type='text/csv')

    except HTTPException as he:
        secure_logger.error(f"Download error: {he.detail}")
        raise he
    except Exception as e:
        secure_logger.error(f"Download error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/data/{file_id}/{data_type}")
async def get_data_preview(file_id: str, data_type: str, limit: int = 50):
    """Get data preview for web display"""

    try:
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")

        if data_type == "original":
            df = file_storage[file_id]['original_data']
        elif data_type == "cleaned":
            if 'cleaned_data' not in file_storage[file_id]:
                raise HTTPException(
                    status_code=404, detail="Cleaned data not available")
            df = file_storage[file_id]['cleaned_data']
        else:
            raise HTTPException(
                status_code=400, detail="Invalid data type. Use 'original' or 'cleaned'")

        # Convert to JSON for frontend display
        preview_df = df.head(limit)

        # Handle NaN values for JSON serialization
        preview_df = preview_df.fillna("")  # Replace NaN with empty strings

        return {
            "data": preview_df.to_dict('records'),
            "columns": list(df.columns),
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }

    except Exception as e:
        secure_logger.error(f"Data preview error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Data preview failed: {str(e)}")


@app.get("/report/{file_id}")
async def get_cleaning_report(file_id: str):
    """Get comprehensive cleaning report with visualizations"""

    try:
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")

        data = file_storage[file_id]

        # Generate comprehensive report
        report = {
            "file_info": {
                "filename": data['filename'],
                "upload_time": data['upload_time']
            },
            "original_analysis": data['analysis'],
            "cleaning_report": data.get('cleaning_report', {}),
            "comparison": {}
        }

        # Add comparison if cleaned data exists
        if 'cleaned_data' in data:
            original_df = data['original_data']
            cleaned_df = data['cleaned_data']

            report["comparison"] = {
                "shape_change": {
                    "before": {"rows": len(original_df), "columns": len(original_df.columns)},
                    "after": {"rows": len(cleaned_df), "columns": len(cleaned_df.columns)}
                },
                "quality_improvement": {
                    "before": data['analysis']['quality_score'],
                    "after": cleaner.analyze_data(cleaned_df)['quality_score']
                }
            }

        return report

    except Exception as e:
        secure_logger.error(f"Report error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Report generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Autonomous AI Data Cleaning Server...")
    uvicorn.run(app, host="0.0.0.0", port=8003)
