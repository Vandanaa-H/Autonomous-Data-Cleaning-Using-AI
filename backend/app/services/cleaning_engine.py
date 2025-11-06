import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Optional

from app.models.schemas import CleaningReport, DetectedIssue, CleaningAction, CleaningStrategy
from app.services.detection.missing_detector import MissingValueDetector
from app.services.detection.duplicate_detector import DuplicateDetector
from app.services.detection.outlier_detector import OutlierDetector
from app.services.detection.inconsistency_detector import InconsistencyDetector
from app.services.cleaning.strategy_selector import StrategySelector
from app.services.cleaning.data_cleaner import DataCleaner
from app.utils.file_utils import detect_file_type

logger = logging.getLogger(__name__)


class CleaningEngine:
    """Main engine for coordinating the data cleaning process"""

    def __init__(self):
        self.detectors = {
            'missing': MissingValueDetector(),
            'duplicates': DuplicateDetector(),
            'outliers': OutlierDetector(),
            'inconsistencies': InconsistencyDetector()
        }
        self.strategy_selector = StrategySelector()
        self.data_cleaner = DataCleaner()

    async def clean_dataset(
        self,
        file_id: str,
        file_path: Path,
        target_task: Optional[str] = None,
        custom_rules: Optional[Dict[str, Any]] = None
    ) -> CleaningReport:
        """
        Main cleaning workflow
        """

        start_time = time.time()

        try:
            logger.info(f"Starting cleaning process for file: {file_id}")

            # Load data
            df = self._load_data(file_path)
            original_df = df.copy()

            # Collect before stats
            before_stats = self._calculate_stats(df)

            # Detect issues
            all_issues = []
            for detector_name, detector in self.detectors.items():
                issues = detector.detect(df)
                all_issues.extend(issues)
                logger.info(
                    f"{detector_name} detector found {len(issues)} issues")

            # Clean data step by step
            actions_taken = []
            for issue in all_issues:

                # Select best strategy for this issue
                strategies = self.strategy_selector.get_strategies_for_issue(
                    issue, target_task)
                best_strategy = self.strategy_selector.select_best_strategy(
                    df, issue, strategies, target_task
                )

                # Apply cleaning strategy
                df, result = self.data_cleaner.apply_strategy(
                    df, issue, best_strategy)

                # Record action
                action = CleaningAction(
                    issue=issue,
                    strategy=best_strategy,
                    result=result
                )
                actions_taken.append(action)

                logger.info(
                    f"Applied {best_strategy.strategy_name} for {issue.issue_type}")

            # Calculate after stats
            after_stats = self._calculate_stats(df)

            # Save cleaned data
            cleaned_file_path = self._save_cleaned_data(file_id, df, file_path)

            # Create report
            processing_time = time.time() - start_time

            report = CleaningReport(
                file_id=file_id,
                original_file=file_path.name,
                cleaned_file=cleaned_file_path.name,
                processing_time=processing_time,
                issues_detected=all_issues,
                actions_taken=actions_taken,
                before_stats=before_stats,
                after_stats=after_stats
            )

            # Save report
            self._save_report(file_id, report)

            logger.info(
                f"Cleaning completed for {file_id} in {processing_time:.2f} seconds")
            return report

        except Exception as e:
            logger.error(f"Cleaning failed for {file_id}: {str(e)}")
            raise

    def _load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from file"""

        file_type = detect_file_type(file_path.name)

        if file_type.value == "csv":
            return pd.read_csv(file_path)
        elif file_type.value == "excel":
            return pd.read_excel(file_path)
        elif file_type.value == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(
                f"Unsupported file type for cleaning: {file_type}")

    def _calculate_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate dataset statistics"""

        total_rows = int(len(df))
        total_cols = int(len(df.columns))
        missing_total = int(df.isnull().sum().sum())
        duplicates = int(len(df) - len(df.drop_duplicates()))
        num_cols = int(len(df.select_dtypes(include=[np.number]).columns))
        text_cols = int(len(df.select_dtypes(include=[object]).columns))
        mem_usage = int(df.memory_usage(deep=True).sum())

        return {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'missing_values': missing_total,
            'duplicates': duplicates,
            'numeric_columns': num_cols,
            'text_columns': text_cols,
            'memory_usage': mem_usage
        }

    def _save_cleaned_data(self, file_id: str, df: pd.DataFrame, original_path: Path) -> Path:
        """Save cleaned dataset"""

        from app.core.config import settings

        # Determine output format based on original file
        extension = original_path.suffix.lower()
        cleaned_file_path = settings.OUTPUT_DIR / \
            f"{file_id}_cleaned{extension}"

        if extension == '.csv':
            df.to_csv(cleaned_file_path, index=False)
        elif extension in ['.xlsx', '.xls']:
            df.to_excel(cleaned_file_path, index=False)
        elif extension == '.json':
            df.to_json(cleaned_file_path, orient='records', indent=2)

        return cleaned_file_path

    def _save_report(self, file_id: str, report: CleaningReport):
        """Save cleaning report"""

        from app.core.config import settings
        import json

        # Save JSON report
        report_path = settings.OUTPUT_DIR / f"{file_id}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)

        # Generate HTML report
        html_report = self._generate_html_report(report)
        html_path = settings.OUTPUT_DIR / f"{file_id}_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)

    def _generate_html_report(self, report: CleaningReport) -> str:
        """Generate HTML version of the report"""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Cleaning Report - {report.file_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .issue {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .action {{ background-color: #d4edda; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Cleaning Report</h1>
                <p><strong>File ID:</strong> {report.file_id}</p>
                <p><strong>Original File:</strong> {report.original_file}</p>
                <p><strong>Cleaned File:</strong> {report.cleaned_file}</p>
                <p><strong>Processing Time:</strong> {report.processing_time:.2f} seconds</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Before</th><th>After</th><th>Improvement</th></tr>
                    <tr>
                        <td>Total Rows</td>
                        <td>{report.before_stats.get('total_rows', 0)}</td>
                        <td>{report.after_stats.get('total_rows', 0)}</td>
                        <td>{report.after_stats.get('total_rows', 0) - report.before_stats.get('total_rows', 0)}</td>
                    </tr>
                    <tr>
                        <td>Missing Values</td>
                        <td>{report.before_stats.get('missing_values', 0)}</td>
                        <td>{report.after_stats.get('missing_values', 0)}</td>
                        <td>{report.before_stats.get('missing_values', 0) - report.after_stats.get('missing_values', 0)}</td>
                    </tr>
                    <tr>
                        <td>Duplicates</td>
                        <td>{report.before_stats.get('duplicates', 0)}</td>
                        <td>{report.after_stats.get('duplicates', 0)}</td>
                        <td>{report.before_stats.get('duplicates', 0) - report.after_stats.get('duplicates', 0)}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Issues Detected ({len(report.issues_detected)})</h2>
                {''.join([f'<div class="issue"><strong>{issue.issue_type}</strong> - {issue.description} (Severity: {issue.severity})</div>' for issue in report.issues_detected])}
            </div>
            
            <div class="section">
                <h2>Actions Taken ({len(report.actions_taken)})</h2>
                {''.join([f'<div class="action"><strong>{action.strategy.strategy_name}</strong> - {action.strategy.description}</div>' for action in report.actions_taken])}
            </div>
        </body>
        </html>
        """

        return html
