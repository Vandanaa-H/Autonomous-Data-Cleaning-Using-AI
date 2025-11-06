import pandas as pd
import numpy as np
import re
from typing import List
from app.models.schemas import DetectedIssue


class InconsistencyDetector:
    """Detector for data inconsistencies and format issues"""

    def detect(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect various types of inconsistencies"""

        issues = []

        # Detect format inconsistencies
        issues.extend(self._detect_date_format_issues(df))
        issues.extend(self._detect_numeric_format_issues(df))
        issues.extend(self._detect_text_inconsistencies(df))
        issues.extend(self._detect_case_inconsistencies(df))

        return issues

    def _detect_date_format_issues(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect inconsistent date formats"""

        issues = []

        for column in df.select_dtypes(include=['object']).columns:
            # Sample some values to check for date patterns
            sample_values = df[column].dropna().astype(str).head(100)

            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'\d{2}\.\d{2}\.\d{4}'  # MM.DD.YYYY
            ]

            pattern_matches = {}
            for pattern in date_patterns:
                matches = sample_values.str.contains(
                    pattern, regex=True, na=False).sum()
                if matches > 0:
                    pattern_matches[pattern] = matches

            # If multiple date patterns found in same column
            if len(pattern_matches) > 1:
                issue = DetectedIssue(
                    issue_type="inconsistent_date_format",
                    column=column,
                    description=f"Column '{column}' contains multiple date formats",
                    severity="medium",
                    affected_rows=sum(pattern_matches.values())
                )
                issues.append(issue)

        return issues

    def _detect_numeric_format_issues(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect inconsistent numeric formats"""

        issues = []

        for column in df.select_dtypes(include=['object']).columns:
            sample_values = df[column].dropna().astype(str).head(100)

            # Check for numbers with different decimal separators
            comma_decimals = sample_values.str.contains(
                r'\d+,\d+', regex=True, na=False).sum()
            dot_decimals = sample_values.str.contains(
                r'\d+\.\d+', regex=True, na=False).sum()

            if comma_decimals > 0 and dot_decimals > 0:
                issue = DetectedIssue(
                    issue_type="inconsistent_decimal_separator",
                    column=column,
                    description=f"Column '{column}' uses both comma and dot as decimal separators",
                    severity="medium",
                    affected_rows=comma_decimals + dot_decimals
                )
                issues.append(issue)

            # Check for numbers with currency symbols
            currency_symbols = sample_values.str.contains(
                r'[\$€£¥]', regex=True, na=False).sum()
            if currency_symbols > 0:
                issue = DetectedIssue(
                    issue_type="currency_in_numeric",
                    column=column,
                    description=f"Column '{column}' contains currency symbols in {currency_symbols} values",
                    severity="low",
                    affected_rows=currency_symbols
                )
                issues.append(issue)

        return issues

    def _detect_text_inconsistencies(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect text-based inconsistencies"""

        issues = []

        for column in df.select_dtypes(include=['object']).columns:

            # Check for leading/trailing whitespace
            values_with_whitespace = df[column].astype(
                str).str.strip() != df[column].astype(str)
            whitespace_count = int(values_with_whitespace.sum())

            if whitespace_count > 0:
                issue = DetectedIssue(
                    issue_type="leading_trailing_whitespace",
                    column=column,
                    description=f"Column '{column}' has {whitespace_count} values with leading/trailing whitespace",
                    severity="low",
                    affected_rows=int(whitespace_count)
                )
                issues.append(issue)

            # Check for inconsistent encoding/special characters
            non_ascii_count = int(df[column].astype(str).str.contains(
                r'[^\x00-\x7F]', regex=True, na=False).sum())
            if non_ascii_count > 0:
                issue = DetectedIssue(
                    issue_type="non_ascii_characters",
                    column=column,
                    description=f"Column '{column}' contains {non_ascii_count} values with non-ASCII characters",
                    severity="low",
                    affected_rows=int(non_ascii_count)
                )
                issues.append(issue)

        return issues

    def _detect_case_inconsistencies(self, df: pd.DataFrame) -> List[DetectedIssue]:
        """Detect inconsistent case usage"""

        issues = []

        for column in df.select_dtypes(include=['object']).columns:
            sample_values = df[column].dropna().astype(str)

            if len(sample_values) == 0:
                continue

            # Check for mixed case in what should be consistent categories
            unique_values = sample_values.unique()

            # Group by lowercase version and see if there are multiple variations
            case_groups = {}
            for value in unique_values:
                lower_key = value.lower()
                if lower_key not in case_groups:
                    case_groups[lower_key] = []
                case_groups[lower_key].append(value)

            # Count groups with multiple case variations
            inconsistent_cases = int(
                sum(1 for group in case_groups.values() if len(group) > 1))

            if inconsistent_cases > 0:
                affected_rows = int(
                    sum(len(group) for group in case_groups.values() if len(group) > 1))

                issue = DetectedIssue(
                    issue_type="inconsistent_case",
                    column=column,
                    description=f"Column '{column}' has {inconsistent_cases} values with inconsistent case",
                    severity="low",
                    affected_rows=affected_rows
                )
                issues.append(issue)

        return issues
