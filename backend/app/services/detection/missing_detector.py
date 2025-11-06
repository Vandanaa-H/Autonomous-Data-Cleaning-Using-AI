import pandas as pd
import numpy as np
from typing import List
from app.models.schemas import DetectedIssue


class MissingValueDetector:
    """Detector for missing values in dataset"""

    def detect(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect missing values in all columns"""

        issues = []

        for column in df.columns:
            missing_count = int(df[column].isnull().sum())

            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100

                # Determine severity
                if missing_percentage > 50:
                    severity = "high"
                elif missing_percentage > 20:
                    severity = "medium"
                else:
                    severity = "low"

                issue = DetectedIssue(
                    issue_type="missing_values",
                    column=column,
                    description=f"Column '{column}' has {missing_count} missing values ({missing_percentage:.1f}%)",
                    severity=severity,
                    affected_rows=int(missing_count)
                )
                issues.append(issue)

        return issues

    def detect_patterns(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect patterns in missing values"""

        issues = []

        # Check for columns that are completely empty
        empty_columns = df.columns[df.isnull().all()].tolist()
        for col in empty_columns:
            issue = DetectedIssue(
                issue_type="empty_column",
                column=col,
                description=f"Column '{col}' is completely empty",
                severity="high",
                affected_rows=len(df)
            )
            issues.append(issue)

        # Check for rows that are mostly empty
        row_missing_percentage = df.isnull().sum(axis=1) / len(df.columns) * 100
        mostly_empty_rows = (row_missing_percentage > 80).sum()

        if mostly_empty_rows > 0:
            issue = DetectedIssue(
                issue_type="mostly_empty_rows",
                description=f"{mostly_empty_rows} rows have more than 80% missing values",
                severity="medium",
                affected_rows=mostly_empty_rows
            )
            issues.append(issue)

        return issues
