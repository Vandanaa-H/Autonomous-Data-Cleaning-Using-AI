import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import logging
from pathlib import Path

from app.models.schemas import DetectedIssue, CleaningStrategy, FileType

logger = logging.getLogger(__name__)


class DataCleaner:
    """Applies cleaning strategies to datasets"""

    def apply_strategy(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        strategy: CleaningStrategy
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply a cleaning strategy to the dataset"""

        strategy_name = strategy.strategy_name
        result = {"strategy_applied": strategy_name,
                  "rows_affected": 0, "details": {}}

        try:
            if strategy_name == "drop_rows":
                df, rows_affected = self._drop_rows_with_missing(
                    df, issue, strategy.parameters)
                result["rows_affected"] = int(rows_affected)

            elif strategy_name in ["mean_imputation", "median_imputation", "mode_imputation"]:
                df, rows_affected = self._simple_imputation(
                    df, issue, strategy)
                result["rows_affected"] = int(rows_affected)

            elif strategy_name == "knn_imputation":
                df, rows_affected = self._knn_imputation(
                    df, issue, strategy.parameters)
                result["rows_affected"] = int(rows_affected)

            elif strategy_name == "iterative_imputation":
                df, rows_affected = self._iterative_imputation(
                    df, issue, strategy.parameters)
                result["rows_affected"] = int(rows_affected)

            elif strategy_name == "categorical_context_imputation":
                df, rows_affected = self._categorical_context_imputation(
                    df, issue, strategy.parameters)
                result["rows_affected"] = int(rows_affected)

            elif strategy_name == "drop_duplicates":
                df, rows_affected = self._drop_duplicates(
                    df, strategy.parameters)
                result["rows_affected"] = int(rows_affected)

            elif strategy_name == "remove_outliers":
                df, rows_affected = self._remove_outliers(
                    df, issue, strategy.parameters)
                result["rows_affected"] = int(rows_affected)

            elif strategy_name == "cap_outliers":
                df, rows_affected = self._cap_outliers(
                    df, issue, strategy.parameters)
                result["rows_affected"] = int(rows_affected)

            elif strategy_name == "transform_outliers":
                df, rows_affected = self._transform_outliers(
                    df, issue, strategy.parameters)
                result["rows_affected"] = int(rows_affected)

            elif strategy_name in ["lowercase", "uppercase", "title_case"]:
                df, rows_affected = self._fix_case_inconsistency(
                    df, issue, strategy_name)
                result["rows_affected"] = int(rows_affected)

            elif strategy_name == "strip_whitespace":
                df, rows_affected = self._strip_whitespace(df, issue)
                result["rows_affected"] = int(rows_affected)

            else:
                logger.warning(f"Unknown strategy: {strategy_name}")

            result["success"] = True
            logger.info(
                f"Applied {strategy_name} to {result['rows_affected']} rows")

        except Exception as e:
            logger.error(f"Failed to apply strategy {strategy_name}: {str(e)}")
            result["success"] = False
            result["error"] = str(e)

        return df, result

    def _categorical_context_imputation(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        parameters: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, int]:
        """Impute missing values in a categorical column using group-wise mode with fallback.

        - Pick up to N parent categorical columns (fewest unique values, exclude target),
          bounded by max_categories and missing_ratio <= 0.5.
        - Fill by group-wise mode; fallback to global mode for any remaining.
        """

        target = issue.column
        if not target or target not in df.columns:
            return df, 0

        # Only for categorical-like columns
        if not (df[target].dtype == 'object' or str(df[target].dtype).startswith('category')):
            return df, 0

        mask_missing = df[target].isna() | (
            df[target].astype(str).str.strip() == '')
        missing_total = int(mask_missing.sum())
        if missing_total == 0:
            return df, 0

        max_categories = int(parameters.get('max_categories', 30))
        max_parents = int(parameters.get('max_parents', 2))

        # Candidate parent columns
        candidate_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if col == target:
                continue
            nunique = int(df[col].nunique(dropna=True))
            if 1 < nunique <= max_categories:
                miss_ratio = float(df[col].isna().mean())
                if miss_ratio <= 0.5:
                    candidate_cols.append((col, nunique))

        candidate_cols.sort(key=lambda x: x[1])
        parents = [c for c, _ in candidate_cols[:max_parents]]

        # Fallback to global mode if no parents
        if not parents:
            mode_vals = df[target].mode(dropna=True)
            if len(mode_vals) == 0:
                return df, 0
            df.loc[mask_missing, target] = mode_vals.iloc[0]
            return df, missing_total

        non_null = df.loc[~df[target].isna() & (
            df[target].astype(str).str.strip() != '')]
        if non_null.empty:
            mode_vals = df[target].mode(dropna=True)
            if len(mode_vals) == 0:
                return df, 0
            df.loc[mask_missing, target] = mode_vals.iloc[0]
            return df, missing_total

        def make_key(frame: pd.DataFrame) -> pd.Series:
            return frame[parents].astype(str).agg('||'.join, axis=1)

        keys_non_null = make_key(non_null)
        mapping: Dict[str, Any] = {}
        grouped = non_null.groupby(keys_non_null)[target]
        for k, series in grouped:
            m = series.mode(dropna=True)
            if len(m) > 0:
                mapping[k] = m.iloc[0]

        if not mapping:
            mode_vals = df[target].mode(dropna=True)
            if len(mode_vals) == 0:
                return df, 0
            df.loc[mask_missing, target] = mode_vals.iloc[0]
            return df, missing_total

        keys_all = make_key(df)
        filled = 0
        to_fill_idx = df.index[mask_missing]
        for idx in to_fill_idx:
            key = keys_all.loc[idx]
            if key in mapping:
                df.at[idx, target] = mapping[key]
                filled += 1

        # Fill remaining with global mode
        remaining_mask = df[target].isna() | (
            df[target].astype(str).str.strip() == '')
        remaining = int(remaining_mask.sum())
        if remaining > 0:
            mode_vals = df[target].mode(dropna=True)
            if len(mode_vals) > 0:
                df.loc[remaining_mask, target] = mode_vals.iloc[0]
                filled = int(filled + remaining)

        return df, int(filled)

    def _drop_rows_with_missing(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        parameters: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, int]:
        """Drop rows with missing values"""

        initial_rows = len(df)

        if issue.column:
            # Drop rows where specific column is missing
            df_cleaned = df.dropna(subset=[issue.column])
        else:
            # Drop rows with any missing values
            threshold = parameters.get("threshold", 0.5)
            df_cleaned = df.dropna(thresh=int(len(df.columns) * threshold))

        rows_affected = initial_rows - len(df_cleaned)
        return df_cleaned, rows_affected

    def _simple_imputation(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        strategy: CleaningStrategy
    ) -> Tuple[pd.DataFrame, int]:
        """Apply simple imputation strategies"""

        column = issue.column
        if not column:
            return df, 0

        initial_nulls = df[column].isnull().sum()

        if strategy.strategy_name == "mean_imputation":
            if df[column].dtype in ['int64', 'float64']:
                fill_value = df[column].mean()
            else:
                return df, 0  # Can't calculate mean for non-numeric

        elif strategy.strategy_name == "median_imputation":
            if df[column].dtype in ['int64', 'float64']:
                fill_value = df[column].median()
            else:
                return df, 0

        elif strategy.strategy_name == "mode_imputation":
            mode_values = df[column].mode()
            if len(mode_values) > 0:
                fill_value = mode_values.iloc[0]
            else:
                return df, 0

        df[column] = df[column].fillna(fill_value)
        rows_affected = initial_nulls

        return df, rows_affected

    def _knn_imputation(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        parameters: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, int]:
        """Apply KNN imputation"""

        column = issue.column
        if not column:
            return df, 0

        initial_nulls = df[column].isnull().sum()

        # Only apply to numeric columns
        numeric_columns = df.select_dtypes(
            include=[np.number]).columns.tolist()
        if column not in numeric_columns:
            return df, 0

        n_neighbors = parameters.get("n_neighbors", 5)

        # Apply KNN imputation to numeric columns only
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_numeric = df[numeric_columns].copy()
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_numeric),
            columns=numeric_columns,
            index=df.index
        )

        # Update the original dataframe
        for col in numeric_columns:
            df[col] = df_imputed[col]

        return df, initial_nulls

    def _iterative_imputation(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        parameters: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, int]:
        """Apply iterative imputation"""

        column = issue.column
        if not column:
            return df, 0

        initial_nulls = df[column].isnull().sum()

        # Only apply to numeric columns
        numeric_columns = df.select_dtypes(
            include=[np.number]).columns.tolist()
        if column not in numeric_columns:
            return df, 0

        max_iter = parameters.get("max_iter", 10)

        # Apply iterative imputation
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        df_numeric = df[numeric_columns].copy()
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_numeric),
            columns=numeric_columns,
            index=df.index
        )

        # Update the original dataframe
        for col in numeric_columns:
            df[col] = df_imputed[col]

        return df, initial_nulls

    def _drop_duplicates(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate rows"""

        initial_rows = len(df)
        keep_strategy = parameters.get("keep", "first")

        df_cleaned = df.drop_duplicates(keep=keep_strategy)
        rows_affected = initial_rows - len(df_cleaned)

        return df_cleaned, rows_affected

    def _remove_outliers(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        parameters: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, int]:
        """Remove outlier rows"""

        column = issue.column
        if not column or df[column].dtype not in ['int64', 'float64']:
            return df, 0

        initial_rows = len(df)
        method = parameters.get("method", "iqr")

        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_cleaned = df[(df[column] >= lower_bound) &
                            (df[column] <= upper_bound)]

        rows_affected = initial_rows - len(df_cleaned)
        return df_cleaned, rows_affected

    def _cap_outliers(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        parameters: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, int]:
        """Cap outliers to acceptable range"""

        column = issue.column
        if not column or df[column].dtype not in ['int64', 'float64']:
            return df, 0

        method = parameters.get("method", "percentile")

        if method == "percentile":
            lower_percentile = parameters.get("lower", 0.05)
            upper_percentile = parameters.get("upper", 0.95)

            lower_bound = df[column].quantile(lower_percentile)
            upper_bound = df[column].quantile(upper_percentile)

            original_values = df[column].copy()
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

            rows_affected = (original_values != df[column]).sum()

        return df, rows_affected

    def _transform_outliers(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        parameters: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, int]:
        """Apply transformation to reduce outlier impact"""

        column = issue.column
        if not column or df[column].dtype not in ['int64', 'float64']:
            return df, 0

        method = parameters.get("method", "log")

        if method == "log":
            # Apply log transformation (add 1 to handle zeros)
            min_val = df[column].min()
            if min_val <= 0:
                df[column] = np.log1p(df[column] - min_val + 1)
            else:
                df[column] = np.log1p(df[column])

        # All values are affected by transformation
        rows_affected = len(df[column].dropna())

        return df, rows_affected

    def _fix_case_inconsistency(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        strategy_name: str
    ) -> Tuple[pd.DataFrame, int]:
        """Fix case inconsistencies in text data"""

        column = issue.column
        if not column:
            return df, 0

        original_values = df[column].astype(str).copy()

        if strategy_name == "lowercase":
            df[column] = df[column].astype(str).str.lower()
        elif strategy_name == "uppercase":
            df[column] = df[column].astype(str).str.upper()
        elif strategy_name == "title_case":
            df[column] = df[column].astype(str).str.title()

        rows_affected = (original_values != df[column].astype(str)).sum()

        return df, rows_affected

    def _strip_whitespace(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue
    ) -> Tuple[pd.DataFrame, int]:
        """Remove leading and trailing whitespace"""

        column = issue.column
        if not column:
            return df, 0

        original_values = df[column].astype(str).copy()
        df[column] = df[column].astype(str).str.strip()

        rows_affected = (original_values != df[column].astype(str)).sum()

        return df, rows_affected

    def process_unstructured_file(self, file_path: Path, file_type: FileType) -> pd.DataFrame:
        """Process unstructured files and convert to structured DataFrame"""

        try:
            from app.services.nlp.unstructured_processor import get_unstructured_processor

            processor = get_unstructured_processor()

            if not processor.can_process(file_type):
                raise ValueError(f"Cannot process file type: {file_type}")

            # Process the file
            df = processor.process_file(file_path, file_type)

            # Convert to more tabular format if possible
            if 'text' in df.columns:
                df = processor.convert_to_tabular(df, text_column='text')

            logger.info(
                f"Successfully processed unstructured file: {file_path}")
            logger.info(f"Resulting DataFrame shape: {df.shape}")

            return df

        except Exception as e:
            logger.error(
                f"Error processing unstructured file {file_path}: {str(e)}")
            raise

    def apply_nlp_cleaning(self, df: pd.DataFrame, text_columns: Optional[list] = None) -> pd.DataFrame:
        """Apply NLP-based cleaning to text columns"""

        try:
            from app.services.nlp.nlp_processor import get_nlp_processor

            nlp_processor = get_nlp_processor()

            if not nlp_processor.initialized:
                logger.warning(
                    "NLP processor not initialized - skipping NLP cleaning")
                return df

            # Auto-detect text columns if not provided
            if text_columns is None:
                text_columns = df.select_dtypes(
                    include=['object']).columns.tolist()

            df_cleaned = df.copy()

            for column in text_columns:
                if column not in df_cleaned.columns:
                    continue

                logger.info(f"Applying NLP cleaning to column: {column}")

                # Apply spell correction
                if hasattr(nlp_processor, 'correct_spelling'):
                    df_cleaned[column] = df_cleaned[column].astype(str).apply(
                        lambda x: nlp_processor.correct_spelling(
                            x) if pd.notna(x) and x.strip() else x
                    )

                # Detect and handle text anomalies
                if hasattr(nlp_processor, 'detect_text_anomalies'):
                    anomalies = nlp_processor.detect_text_anomalies(
                        df_cleaned[column].tolist())
                    if anomalies:
                        logger.info(
                            f"Found {len(anomalies)} text anomalies in column {column}")
                        # Log anomalies for review
                        for idx, anomaly in anomalies[:5]:  # Log first 5
                            logger.info(
                                f"Text anomaly at index {idx}: {anomaly}")

            return df_cleaned

        except Exception as e:
            logger.error(f"Error applying NLP cleaning: {str(e)}")
            return df
