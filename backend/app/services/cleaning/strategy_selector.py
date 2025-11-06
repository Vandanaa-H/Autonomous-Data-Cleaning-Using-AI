import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from app.models.schemas import DetectedIssue, CleaningStrategy


class StrategySelector:
    """Intelligent strategy selector that chooses the best cleaning approach"""

    def __init__(self):
        self.strategy_catalog = self._build_strategy_catalog()

    def get_strategies_for_issue(self, issue: DetectedIssue, target_task: Optional[str] = None) -> List[CleaningStrategy]:
        """Get candidate strategies for a specific issue"""

        issue_type = issue.issue_type

        if issue_type not in self.strategy_catalog:
            return []

        strategies = self.strategy_catalog[issue_type].copy()

        # Filter strategies based on target task if specified
        if target_task:
            strategies = self._filter_by_target_task(strategies, target_task)

        return strategies

    def select_best_strategy(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        candidate_strategies: List[CleaningStrategy],
        target_task: Optional[str] = None
    ) -> CleaningStrategy:
        """Select the best strategy using evaluation metrics"""

        if len(candidate_strategies) == 1:
            return candidate_strategies[0]

        if len(candidate_strategies) == 0:
            # Return a default strategy
            return CleaningStrategy(
                strategy_name="no_action",
                description="No cleaning action applied"
            )

        # If we have target task information, use ML-based evaluation
        if target_task and self._can_evaluate_with_ml(df):
            return self._select_with_ml_evaluation(df, issue, candidate_strategies, target_task)
        else:
            # Use proxy metrics for evaluation
            return self._select_with_proxy_metrics(df, issue, candidate_strategies)

    def _build_strategy_catalog(self) -> Dict[str, List[CleaningStrategy]]:
        """Build catalog of available cleaning strategies"""

        return {
            "missing_values": [
                CleaningStrategy(
                    strategy_name="drop_rows",
                    description="Remove rows with missing values",
                    parameters={"threshold": 0.5}
                ),
                CleaningStrategy(
                    strategy_name="mean_imputation",
                    description="Fill missing values with column mean",
                    parameters={"strategy": "mean"}
                ),
                CleaningStrategy(
                    strategy_name="median_imputation",
                    description="Fill missing values with column median",
                    parameters={"strategy": "median"}
                ),
                CleaningStrategy(
                    strategy_name="mode_imputation",
                    description="Fill missing values with column mode",
                    parameters={"strategy": "mode"}
                ),
                CleaningStrategy(
                    strategy_name="categorical_context_imputation",
                    description="Fill categorical missing values using group-wise mode with parent columns",
                    parameters={"max_parents": 2, "max_categories": 30}
                ),
                CleaningStrategy(
                    strategy_name="knn_imputation",
                    description="Fill missing values using KNN imputation",
                    parameters={"n_neighbors": 5}
                ),
                CleaningStrategy(
                    strategy_name="iterative_imputation",
                    description="Fill missing values using iterative imputation",
                    parameters={"max_iter": 10}
                )
            ],
            "exact_duplicates": [
                CleaningStrategy(
                    strategy_name="drop_duplicates",
                    description="Remove exact duplicate rows",
                    parameters={"keep": "first"}
                ),
                CleaningStrategy(
                    strategy_name="keep_duplicates",
                    description="Keep all duplicate rows",
                    parameters={}
                )
            ],
            "numeric_outliers": [
                CleaningStrategy(
                    strategy_name="remove_outliers",
                    description="Remove outlier rows",
                    parameters={"method": "iqr"}
                ),
                CleaningStrategy(
                    strategy_name="cap_outliers",
                    description="Cap outliers to acceptable range",
                    parameters={"method": "percentile",
                                "lower": 0.05, "upper": 0.95}
                ),
                CleaningStrategy(
                    strategy_name="transform_outliers",
                    description="Apply log transformation to reduce outlier impact",
                    parameters={"method": "log"}
                ),
                CleaningStrategy(
                    strategy_name="keep_outliers",
                    description="Keep outliers as they may be important",
                    parameters={}
                )
            ],
            "inconsistent_case": [
                CleaningStrategy(
                    strategy_name="lowercase",
                    description="Convert all text to lowercase",
                    parameters={}
                ),
                CleaningStrategy(
                    strategy_name="uppercase",
                    description="Convert all text to uppercase",
                    parameters={}
                ),
                CleaningStrategy(
                    strategy_name="title_case",
                    description="Convert text to title case",
                    parameters={}
                )
            ],
            "leading_trailing_whitespace": [
                CleaningStrategy(
                    strategy_name="strip_whitespace",
                    description="Remove leading and trailing whitespace",
                    parameters={}
                )
            ]
        }

    def _filter_by_target_task(self, strategies: List[CleaningStrategy], target_task: str) -> List[CleaningStrategy]:
        """Filter strategies based on target ML task"""

        # For classification tasks, be more conservative with data removal
        if target_task == "classification":
            return [s for s in strategies if "remove" not in s.strategy_name.lower()]

        # For regression tasks, outlier removal might be more beneficial
        if target_task == "regression":
            return strategies  # Keep all strategies

        return strategies

    def _can_evaluate_with_ml(self, df: pd.DataFrame) -> bool:
        """Check if we can perform ML-based evaluation"""

        # Need at least 100 rows and some numeric columns for meaningful ML evaluation
        return len(df) >= 100 and len(df.select_dtypes(include=[np.number]).columns) > 0

    def _select_with_ml_evaluation(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        strategies: List[CleaningStrategy],
        target_task: str
    ) -> CleaningStrategy:
        """Select strategy using ML performance metrics"""

        try:
            from app.services.cleaning.data_cleaner import DataCleaner

            # Get target column for evaluation (use last numeric column as proxy)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return self._select_with_proxy_metrics(df, issue, strategies)

            target_col = numeric_cols[-1]  # Use last numeric column as target
            feature_cols = [col for col in numeric_cols if col != target_col]

            if len(feature_cols) == 0:
                return self._select_with_proxy_metrics(df, issue, strategies)

            best_strategy = strategies[0]
            best_score = -np.inf

            cleaner = DataCleaner()

            for strategy in strategies:
                try:
                    # Apply strategy to a copy of the data
                    df_copy = df.copy()

                    # Apply the specific strategy
                    if strategy.strategy_name == "drop_rows":
                        df_copy = df_copy.dropna()
                    elif "imputation" in strategy.strategy_name:
                        df_copy = self._apply_imputation_strategy(
                            df_copy, strategy, issue.column)
                    elif strategy.strategy_name == "drop_duplicates":
                        df_copy = df_copy.drop_duplicates()
                    elif "outliers" in strategy.strategy_name:
                        df_copy = self._apply_outlier_strategy(
                            df_copy, strategy, issue.column)

                    # Evaluate with ML model
                    score = self._evaluate_with_ml_model(
                        df_copy, target_col, feature_cols, target_task)

                    if score > best_score:
                        best_score = score
                        best_strategy = strategy

                except Exception as e:
                    # If strategy application fails, skip it
                    continue

            return best_strategy

        except Exception:
            # Fall back to proxy metrics if ML evaluation fails
            return self._select_with_proxy_metrics(df, issue, strategies)

    def _apply_imputation_strategy(self, df: pd.DataFrame, strategy: CleaningStrategy, column: str) -> pd.DataFrame:
        """Apply imputation strategy to specific column"""

        if not column or column not in df.columns:
            return df

        if strategy.strategy_name == "mean_imputation":
            if df[column].dtype in ['int64', 'float64']:
                df[column] = df[column].fillna(df[column].mean())
        elif strategy.strategy_name == "median_imputation":
            if df[column].dtype in ['int64', 'float64']:
                df[column] = df[column].fillna(df[column].median())
        elif strategy.strategy_name == "mode_imputation":
            mode_val = df[column].mode()
            if len(mode_val) > 0:
                df[column] = df[column].fillna(mode_val[0])
        elif strategy.strategy_name == "knn_imputation":
            try:
                from sklearn.impute import KNNImputer
                if df[column].dtype in ['int64', 'float64']:
                    imputer = KNNImputer(
                        n_neighbors=strategy.parameters.get('n_neighbors', 5))
                    df[[column]] = imputer.fit_transform(df[[column]])
            except:
                df[column] = df[column].fillna(df[column].median())

        return df

    def _apply_outlier_strategy(self, df: pd.DataFrame, strategy: CleaningStrategy, column: str) -> pd.DataFrame:
        """Apply outlier handling strategy to specific column"""

        if not column or column not in df.columns or df[column].dtype not in ['int64', 'float64']:
            return df

        if strategy.strategy_name == "remove_outliers":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower) & (df[column] <= upper)]

        elif strategy.strategy_name == "cap_outliers":
            lower_pct = strategy.parameters.get('lower', 0.05)
            upper_pct = strategy.parameters.get('upper', 0.95)
            lower = df[column].quantile(lower_pct)
            upper = df[column].quantile(upper_pct)
            df[column] = df[column].clip(lower=lower, upper=upper)

        elif strategy.strategy_name == "transform_outliers":
            # Apply log transformation
            if (df[column] > 0).all():
                df[column] = np.log1p(df[column])

        return df

    def _evaluate_with_ml_model(self, df: pd.DataFrame, target_col: str, feature_cols: List[str], target_task: str) -> float:
        """Evaluate cleaned data with a quick ML model"""

        try:
            # Prepare data
            X = df[feature_cols].select_dtypes(include=[np.number])
            y = df[target_col]

            # Remove rows with missing values in features or target
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]

            if len(X) < 10:  # Need minimum samples
                return 0.0

            if target_task == "classification":
                # Convert to classification problem
                y_median = y.median()
                y_binary = (y > y_median).astype(int)

                model = RandomForestClassifier(
                    n_estimators=10, random_state=42, max_depth=3)
                scores = cross_val_score(model, X, y_binary, cv=min(
                    3, len(X)//5), scoring='accuracy')

            else:  # regression
                model = RandomForestRegressor(
                    n_estimators=10, random_state=42, max_depth=3)
                scores = cross_val_score(model, X, y, cv=min(
                    3, len(X)//5), scoring='neg_mean_squared_error')
                scores = -scores  # Convert to positive values
                # Convert to 0-1 range (higher is better)
                scores = 1 / (1 + scores)

            return float(np.mean(scores))

        except Exception as e:
            return 0.0

    def _select_with_proxy_metrics(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        strategies: List[CleaningStrategy]
    ) -> CleaningStrategy:
        """Select strategy using proxy metrics (data quality indicators)"""

        # Simple heuristic-based selection

        # For missing values, prefer imputation over deletion unless >70% missing
        if issue.issue_type == "missing_values":
            missing_percentage = (issue.affected_rows / len(df)) * 100

            if missing_percentage > 70:
                return next((s for s in strategies if "drop" in s.strategy_name), strategies[0])
            else:
                # Prefer median for numeric, mode for categorical
                if issue.column and df[issue.column].dtype in ['int64', 'float64']:
                    return next((s for s in strategies if "median" in s.strategy_name), strategies[0])
                else:
                    # Prefer contextual imputation if available for categorical
                    ctx = next((s for s in strategies if s.strategy_name ==
                               "categorical_context_imputation"), None)
                    return ctx if ctx else next((s for s in strategies if "mode" in s.strategy_name), strategies[0])

        # For duplicates, usually drop them
        if issue.issue_type == "exact_duplicates":
            return next((s for s in strategies if "drop" in s.strategy_name), strategies[0])

        # For outliers, prefer capping over removal
        if issue.issue_type == "numeric_outliers":
            return next((s for s in strategies if "cap" in s.strategy_name), strategies[0])

        # Default: return first strategy
        return strategies[0]

    def _simulate_strategy_score(
        self,
        df: pd.DataFrame,
        issue: DetectedIssue,
        strategy: CleaningStrategy,
        target_task: str
    ) -> float:
        """Simulate the effect of applying a strategy and return a quality score"""

        # This is a simplified simulation
        # In real implementation, you'd actually apply the strategy and evaluate

        base_score = 0.5

        # Bonus points for strategies that reduce data loss
        if "imputation" in strategy.strategy_name:
            base_score += 0.2

        if "drop" in strategy.strategy_name:
            base_score -= 0.1

        # Task-specific adjustments
        if target_task == "classification" and "remove_outliers" in strategy.strategy_name:
            base_score -= 0.1  # May remove important decision boundary points

        if target_task == "regression" and "cap_outliers" in strategy.strategy_name:
            base_score += 0.1  # Often beneficial for regression

        return max(0, min(1, base_score))
