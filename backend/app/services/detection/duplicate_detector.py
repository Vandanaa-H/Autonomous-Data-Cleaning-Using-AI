import pandas as pd
import numpy as np
from typing import List
from app.models.schemas import DetectedIssue


class DuplicateDetector:
    """Detector for duplicate records in dataset"""

    def detect(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect exact and near-duplicate records"""

        issues = []

        # Detect exact duplicates
        exact_duplicates = int(df.duplicated().sum())
        if exact_duplicates > 0:
            duplicate_percentage = (exact_duplicates / len(df)) * 100

            severity = "high" if duplicate_percentage > 10 else "medium" if duplicate_percentage > 5 else "low"

            issue = DetectedIssue(
                issue_type="exact_duplicates",
                description=f"Found {exact_duplicates} exact duplicate rows ({duplicate_percentage:.1f}%)",
                severity=severity,
                affected_rows=exact_duplicates
            )
            issues.append(issue)

        # Detect duplicates in specific columns (potential keys)
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].dtype.name.startswith('int'):
                duplicate_values = int(df[column].duplicated().sum())
                # Not mostly duplicates
                if duplicate_values > 0 and duplicate_values < len(df) * 0.8:
                    issue = DetectedIssue(
                        issue_type="column_duplicates",
                        column=column,
                        description=f"Column '{column}' has {duplicate_values} duplicate values",
                        severity="low",
                        affected_rows=int(duplicate_values)
                    )
                    issues.append(issue)

        return issues

    def detect_semantic_duplicates(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect semantic duplicates using text similarity (for text columns)"""

        issues = []

        # This would require more sophisticated NLP processing
        # For now, detect potential semantic duplicates based on similar patterns

        text_columns = df.select_dtypes(include=['object']).columns

        for column in text_columns:
            # Simple approach: detect entries that are very similar after normalization
            normalized = df[column].astype(str).str.lower().str.strip()

            # Group by normalized values and find groups with size > 1
            duplicated_groups = normalized.value_counts()
            potential_semantic_duplicates = duplicated_groups[duplicated_groups > 1].sum(
            ) - len(duplicated_groups[duplicated_groups > 1])

            if potential_semantic_duplicates > 0:
                issue = DetectedIssue(
                    issue_type="potential_semantic_duplicates",
                    column=column,
                    description=f"Column '{column}' may have {potential_semantic_duplicates} semantic duplicates",
                    severity="low",
                    affected_rows=potential_semantic_duplicates
                )
                issues.append(issue)

        return issues
