import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any, List

from app.models.schemas import DataProfile, FileType
from app.utils.file_utils import detect_file_type

logger = logging.getLogger(__name__)


class ProfileService:
    """Service for generating data profiles"""

    def __init__(self):
        self.supported_formats = {
            FileType.CSV: self._load_csv,
            FileType.EXCEL: self._load_excel,
            FileType.JSON: self._load_json,
            FileType.TEXT: self._load_text,
            FileType.PDF: self._load_unstructured,
            FileType.IMAGE: self._load_unstructured
        }

    async def generate_profile(self, file_id: str, file_path: Path) -> DataProfile:
        """Generate comprehensive data profile"""

        try:
            # Detect file type
            file_type = detect_file_type(file_path.name)

            # Load data
            if file_type not in self.supported_formats:
                raise ValueError(
                    f"Unsupported file type for profiling: {file_type}")

            if file_type in [FileType.PDF, FileType.IMAGE]:
                df = self._load_unstructured(file_path, file_type)
            else:
                df = self.supported_formats[file_type](file_path)

            # Generate profile
            profile = DataProfile(
                file_id=file_id,
                total_rows=len(df),
                total_columns=len(df.columns),
                column_info=self._analyze_columns(df),
                missing_values=self._count_missing_values(df),
                duplicates=self._count_duplicates(df),
                outliers=self._detect_outliers(df),
                data_types=self._get_data_types(df)
            )

            # Save profile
            self._save_profile(file_id, profile)

            logger.info(
                f"Profile generated for {file_id}: {profile.total_rows} rows, {profile.total_columns} columns")
            return profile

        except Exception as e:
            logger.error(f"Profile generation failed for {file_id}: {str(e)}")
            raise

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with robust fallbacks for encoding and bad lines"""
        # Try UTF-8 first with safe options
        try:
            return pd.read_csv(
                file_path,
                encoding="utf-8",
                low_memory=False,
                on_bad_lines="skip"  # pandas >= 1.3
            )
        except TypeError:
            # on_bad_lines not supported (older pandas)
            try:
                return pd.read_csv(
                    file_path,
                    encoding="utf-8",
                    low_memory=False,
                    error_bad_lines=False,  # deprecated but works on old pandas
                    warn_bad_lines=False,
                )
            except Exception:
                pass
        except UnicodeDecodeError:
            # Fallback to latin1 when utf-8 fails
            try:
                return pd.read_csv(
                    file_path,
                    encoding="latin1",
                    low_memory=False,
                    on_bad_lines="skip",
                )
            except Exception:
                pass
        # Last resort: try very permissive mode
        return pd.read_csv(file_path, engine="python", low_memory=False)

    def _load_excel(self, file_path: Path) -> pd.DataFrame:
        """Load Excel file with graceful fallback if engine missing"""
        try:
            return pd.read_excel(file_path)
        except ImportError as e:
            # Provide clearer guidance when openpyxl/xlrd is missing
            raise RuntimeError(
                "Excel support requires 'openpyxl' (for .xlsx) or 'xlrd<=1.2.0' (for .xls). Please install the dependency or upload CSV instead."
            ) from e

    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Load JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")

    def _load_text(self, file_path: Path) -> pd.DataFrame:
        """Load text file as single column"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        return pd.DataFrame({'text': [line.strip() for line in lines]})

    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze each column"""

        column_info = {}

        for col in df.columns:
            nunique = int(df[col].nunique())
            null_count = int(df[col].isnull().sum())
            null_pct = float(
                (df[col].isnull().sum() / len(df)) * 100) if len(df) else 0.0

            info = {
                'dtype': str(df[col].dtype),
                'unique_values': nunique,
                'null_count': null_count,
                'null_percentage': null_pct
            }

            # Add statistics for numeric columns
            if df[col].dtype in ['int64', 'float64']:
                info.update({
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else None
                })

            # Add sample values
            sample_values = df[col].dropna().unique()[:5].tolist()
            info['sample_values'] = [str(val) for val in sample_values]

            column_info[col] = info

        return column_info

    def _count_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Count missing values per column (cast to built-in ints)"""
        raw = df.isnull().sum().to_dict()
        return {k: int(v) for k, v in raw.items()}

    def _count_duplicates(self, df: pd.DataFrame) -> int:
        """Count duplicate rows"""
        return len(df) - len(df.drop_duplicates())

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using IQR method"""

        outliers = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_count = len(
                df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            outliers[col] = int(outlier_count)

        return outliers

    def _get_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get data types for all columns"""
        return df.dtypes.astype(str).to_dict()

    def _load_unstructured(self, file_path: Path, file_type: FileType = None) -> pd.DataFrame:
        """Load unstructured data files using the unstructured processor"""

        try:
            from app.services.cleaning.data_cleaner import DataCleaner

            # If file_type is not provided, detect it
            if file_type is None:
                file_type = detect_file_type(file_path.name)

            cleaner = DataCleaner()
            df = cleaner.process_unstructured_file(file_path, file_type)

            logger.info(f"Successfully loaded unstructured file: {file_path}")
            logger.info(f"Extracted data shape: {df.shape}")

            return df

        except Exception as e:
            logger.error(
                f"Error loading unstructured file {file_path}: {str(e)}")
            # Fallback: create a simple DataFrame with error info
            return pd.DataFrame({
                'source': [str(file_path)],
                'error': [str(e)],
                'file_type': [file_type.value if file_type else 'unknown']
            })

    def _save_profile(self, file_id: str, profile: DataProfile):
        """Save profile to file"""

        from app.core.config import settings

        profile_path = settings.OUTPUT_DIR / f"{file_id}_profile.json"

        with open(profile_path, 'w') as f:
            json.dump(profile.dict(), f, indent=2, default=str)
