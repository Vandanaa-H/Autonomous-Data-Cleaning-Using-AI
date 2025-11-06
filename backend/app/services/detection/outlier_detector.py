import pandas as pd
import numpy as np
from typing import List
from scipy import stats
from sklearn.ensemble import IsolationForest
from app.models.schemas import DetectedIssue


class OutlierDetector:
    """Detector for outliers in numeric data"""

    def detect(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect outliers using multiple methods"""

        issues = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            # Skip columns with too many nulls
            if df[column].isnull().sum() / len(df) > 0.5:
                continue

            outliers_iqr = self._detect_iqr_outliers(df[column])
            outliers_zscore = self._detect_zscore_outliers(df[column])
            outliers_isolation = self._detect_isolation_outliers(df[[column]])

            # Combine results (conservative approach - take intersection)
            combined_outliers = len(set(outliers_iqr) & set(outliers_zscore))

            if combined_outliers > 0:
                outlier_percentage = (combined_outliers / len(df)) * 100

                severity = "high" if outlier_percentage > 10 else "medium" if outlier_percentage > 5 else "low"

                issue = DetectedIssue(
                    issue_type="numeric_outliers",
                    column=column,
                    description=f"Column '{column}' has {combined_outliers} outliers ({outlier_percentage:.1f}%)",
                    severity=severity,
                    affected_rows=int(combined_outliers)
                )
                issues.append(issue)

        return issues

    def _detect_iqr_outliers(self, series: pd.Series) -> List[int]:
        """Detect outliers using IQR method"""

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_indices = series[(series < lower_bound) | (
            series > upper_bound)].index.tolist()
        return outlier_indices

    def _detect_zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """Detect outliers using Z-score method"""

        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_indices = series.dropna(
        ).iloc[z_scores > threshold].index.tolist()
        return outlier_indices

    def _detect_isolation_outliers(self, df_subset: pd.DataFrame, contamination: float = 0.1) -> List[int]:
        """Detect outliers using Isolation Forest"""

        try:
            # Remove NaN values for this method
            clean_data = df_subset.dropna()

            if len(clean_data) < 10:  # Need minimum data points
                return []

            isolation_forest = IsolationForest(
                contamination=contamination, random_state=42)
            outlier_labels = isolation_forest.fit_predict(clean_data)

            outlier_indices = clean_data[outlier_labels == -1].index.tolist()
            return outlier_indices

        except Exception:
            return []

    def detect_categorical_outliers(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect outliers in categorical data (rare categories)"""

        issues = []
        categorical_columns = df.select_dtypes(include=['object']).columns

        for column in categorical_columns:
            value_counts = df[column].value_counts()
            total_values = len(df[column].dropna())

            # Find categories that appear very rarely (less than 1% of data)
            rare_categories = value_counts[value_counts / total_values < 0.01]

            if len(rare_categories) > 0:
                rare_count = rare_categories.sum()

                issue = DetectedIssue(
                    issue_type="rare_categories",
                    column=column,
                    description=f"Column '{column}' has {len(rare_categories)} rare categories affecting {rare_count} rows",
                    severity="low",
                    affected_rows=rare_count
                )
                issues.append(issue)

        return issues
