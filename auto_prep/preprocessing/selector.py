import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # noqa F401

from ..utils.abstract import FeatureImportanceSelector, NonRequiredStep, Numerical, RequiredStep, Categorical
from ..utils.config import config
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class VarianceFilter(RequiredStep, Numerical):
    """
    Transformer to remove numerical columns with zero variance.

    Attributes:
        dropped_columns (list): List of dropped columns.
    """

    def __init__(self):
        """
        Initializes the transformer with empty list of dropped columns.
        """

        self.dropped_columns = []

    def fit(self, X: pd.DataFrame) -> "VarianceFilter":
        """
        Identifies columns with zero variances and adds to dropped_columns list.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            VarianceAndUniqueFilter: The fitted transformer instance.
        """
        logger.start_operation("Fitting VarianceFilter")
        try:
            zero_variance = X.var() == 0
            self.dropped_columns = X.columns[zero_variance].tolist()
        except Exception as e:
            logger.error(f"Error in VarianceFilter fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the identified columns with zero variance based on the fit method.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        logger.start_operation(
            f"Transforming data by dropping {len(self.dropped_columns)} zero variance columns."
        )
        try:
            X = X.drop(columns=self.dropped_columns, errors="ignore")
        except Exception as e:
            logger.error(f"Error in VarianceFilter transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the data in one step.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        logger.start_operation("Fitting and transforming data with zero variance")
        try:
            self.fit(X)
            X = self.transform(X)
        except Exception as e:
            logger.error(f"Error in VarianceFilter fit_transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "desc": "Removes columns with zero variance.",
        }


class UniqueFilter(RequiredStep, Categorical):
    """
    Transformer to remove categorical columns 100% unique values.

    Attributes:
        dropped_columns (list): List of dropped columns.
    """

    def __init__(self):
        """
        Initializes the transformer with an empty list of dropped columns.
        """
        self.dropped_columns = []

    def fit(self, X: pd.DataFrame) -> "UniqueFilter":
        """
        Identifies categorical columns with 100% unique values.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            UniqueFilter: The fitted transformer instance.
        """
        logger.start_operation("Fitting UniqueFilter")
        try:
            # Select only categorical columns
            cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
            self.dropped_columns = [
                col for col in cat_cols.columns if X[col].nunique() == len(X)
            ]
        except Exception as e:
            logger.error(f"Error in UniqueFilter fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the identified categorical columns with 100% unique values based on the fit method.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        logger.start_operation(
            f"Transforming data UniqueFilter by dropping {len(self.dropped_columns)} columns with unique values"
        )
        try:
            X = X.drop(columns=self.dropped_columns, errors="ignore")
        except Exception as e:
            logger.error(f"Error in UniqueFilter transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the data in one step.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        logger.start_operation(
            "Fitting and transforming categorical data with 100% unique values"
        )
        try:
            self.fit(X)
        except Exception as e:
            logger.error(f"Error in UniqueFilter fit_transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return self.fit(X).transform(X)

    def is_numerical(self) -> bool:
        return False

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "desc": "Removes categorical columns with 100% unique values. ",
        }


class CorrelationFilter(RequiredStep, Numerical):
    """
    Transformer to detect highly correlated features and drop one of them. Pearsons correlation is used.
    Is a required step in preprocessing.

    Attributes:
        threshold (float): Correlation threshold above which features are considered highly correlated.
        dropped_columns (list): List of columns that were dropped due to high correlation.
    """

    def __init__(self, threshold: float = 0.8):
        """
        Initializes the filter with a specified correlation threshold.

        Args:
            threshold (float): Correlation threshold above which features are considered highly correlated.
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        self.threshold = threshold
        self.dropped_columns = []

    def fit(self, X: pd.DataFrame) -> "CorrelationFilter":
        """
        Identifies highly correlated features. Adds the second one from the pair to the list of columns to be dropped.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            CorrelationFilter: The fitted filter instance.
        """
        logger.start_operation(
            f"Fitting CorrelationFilter with threshold {self.threshold}"
        )
        try:
            corr_matrix = X.corr().abs()
            correlated_pairs = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.threshold:
                        correlated_pairs.add(
                            corr_matrix.columns[j]
                        )  # only the second column of the pair is dropped

            self.dropped_columns = list(correlated_pairs)
        except Exception as e:
            logger.error(f"Error in CorrelationFilter fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Drops all features identified as highly correlated with another feature.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with correlated columns removed.
        """
        logger.start_operation(
            f"Transforming data by dropping {len(self.dropped_columns)} highly correlated columns."
        )
        try:
            X = X.drop(columns=self.dropped_columns, errors="ignore")

            if y is not None:
                y_name = y.name if y.name is not None else "y"
                logger.debug(
                    f"Appending target variable '{y_name}' to the transformed data."
                )
                X[y_name] = y
        except Exception as e:
            logger.error(f"Error in CorrelationFilter transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits and transforms the data by removing correlated features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data.
        """
        logger.start_operation(
            f"Fitting and transforming data with correlation threshold {self.threshold}"
        )
        try:
            self.fit(X)
            X = self.transform(X, y)
        except Exception as e:
            logger.error(f"Error in CorrelationFilter fit_transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "desc": f"Removes one column from pairs of columns correlated above correlation threshold: {self.threshold}.",
            "params": {"threshold": self.threshold},
        }


class CorrelationSelector(NonRequiredStep, Numerical):
    """
    Transformer to select correlation_percent% (rounded to whole number) of features that are most correlated with the target variable.

    Attributes:
         selected_columns (list): List of selected columns based on correlation with the target.
    """

    def __init__(self):
        """
        Initializes the transformer with a specified percentage of top correlated features to keep.

        Args:
            k (float): The percentage of features to retain based on their correlation with the target.
        """
        self.k = config.correlation_selectors_settings["k"]
        self.selected_columns = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CorrelationSelector":
        """
        Identifies the top correlation_percent% (rounded to whole value) of features most correlated with the target variable.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            CorrelationSelector: The fitted transformer instance.
        """
        logger.start_operation(
            f"Fitting CorrelationSelector with top {self.correlation_percent}% correlated features."
        )
        try:
            corr_with_target = X.corrwith(y).abs()
            sorted_corr = corr_with_target.sort_values(ascending=False)
            num_top_features = max(1, round(np.ceil(len(sorted_corr) * self.k / 100)))
            self.selected_columns = sorted_corr.head(num_top_features).index.tolist()
        except Exception as e:
            logger.error(f"Error in CorrelationSelector fit: {e}")
            raise e 
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top correlation_percent% of features most correlated with the target variable.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% correlated features.
        """
        logger.start_operation(
            f"Transforming data by selecting {len(self.selected_columns)} most correlated features."
        )
        try:

            X_selected = X[self.selected_columns].copy()

            if y is not None:
                y_name = y.name if y.name is not None else "y"
                logger.debug(
                    f"Appending target variable '{y_name}' to the transformed data."
                )
                X_selected[y_name] = y
        except Exception as e:
            logger.error(f"Error in CorrelationSelector transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X_selected

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits and transforms the data by selecting the top k% most correlated features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame: The transformed data with selected features.
        """
        logger.start_operation(
            f"Fitting and transforming data with top {self.k}% correlated features."
        )
        try:
            self.fit(X, y)
            X = self.transform(X, y)
        except Exception as e:
            logger.error(f"Error in CorrelationSelector fit_transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "desc": f"Selects the top {self.k}% (rounded to whole number) of features most correlated with the target variable. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"k": self.k},
        }


class FeatureImportanceClassificationSelector(FeatureImportanceSelector):
    """
    Transformer to select k% (rounded to whole number) of features
    that are most important according to Random Forest model for classification.

    Attributes:
        k (float): The percentage of top features to keep based on their importance.
        selected_columns (list): List of selected columns based on feature importance.
    """

    def __init__(self):
        """
        Initializes the transformer with a specified percentage of top important features to keep.

        """
        super().__init__()
        self.feature_importances_ = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "FeatureImportanceClassificationSelector":
        """
        Identifies the feature importances according to the Random Forest model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceClassificationSelector: The fitted transformer instance.
        """
        logger.start_operation(
            f"Fitting FeatureImportanceClassificationSelector with top {self.k}% important features."
        )
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            self.feature_importances_ = model.feature_importances_
            total_features = len(self.feature_importances_)
            num_features_to_select = int(np.ceil(total_features * self.k / 100))
            if num_features_to_select == 0:
                num_features_to_select = 1
            indices = np.argsort(self.feature_importances_)[-num_features_to_select:][
                ::-1
            ]
            self.selected_columns = X.columns[indices].tolist()
        except Exception as e:
            logger.error(f"Error in FeatureImportanceClassificationSelector fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top k% of features most important according to the Random Forest model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        logger.start_operation(
            f"Transforming data by selecting {len(self.selected_columns)} most important features."
        )
        try:
            X_selected = X[self.selected_columns].copy()
            if y is not None:
                if isinstance(y, list):
                    y = pd.Series(y)
                y_name = y.name if y.name is not None else "y"
                logger.debug(
                    f"Appending target variable '{y_name}' to the transformed data."
                )
                X_selected[y_name] = y.values
        except Exception as e:
            logger.error(
                f"Error in FeatureImportanceClassificationSelector transform: {e}"
            )
            raise e
        finally:
            logger.end_operation()
        return X_selected

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits and transforms the data by selecting the top k% most important features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.
        """

        logger.start_operation(
            f"Fitting and transforming data with top {self.k}% important features."
        )
        try:
            self.fit(X, y)
            X = self.transform(X, y)
        except Exception as e:
            logger.error(
                f"Error in FeatureImportanceClassificationSelector fit_transform: {e}"
            )
            raise e
        finally:
            logger.end_operation()
        return X

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "desc": f"Selects the top {self.k}% (rounded to whole number) of features most important according to Random Forest model for classification. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"k": self.k},
        }


class FeatureImportanceRegressionSelector(FeatureImportanceSelector):
    """
    Transformer to select k% (rounded to whole number) of features
    that are most important according to Random Forest model for regression.

    Attributes:
        k (float): The percentage of top features to keep based on their importance.
        selected_columns (list): List of selected columns based on feature importance.
    """

    def __init__(self):
        """
        Initializes the transformer with a specified percentage of top important features to keep.

        """
        super().__init__()
        self.feature_importances_ = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "FeatureImportanceRegressionSelector":
        """
        Identifies the feature importances according to the Random Forest model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceRegressionSelector: The fitted transformer instance.
        """
        logger.start_operation(
            f"Fitting FeatureImportanceRegressionSelector with top {self.k}% important features."
        )
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            self.feature_importances_ = model.feature_importances_
            total_features = len(self.feature_importances_)
            num_features_to_select = int(np.ceil(total_features * self.k / 100))
            if num_features_to_select == 0:
                num_features_to_select = 1
            indices = np.argsort(self.feature_importances_)[-num_features_to_select:][
                ::-1
            ]
            self.selected_columns = X.columns[indices].tolist()
        except Exception as e:
            logger.error(f"Error in FeatureImportanceRegressionSelector fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top k% of features most important according to the Random Forest model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        logger.start_operation(
            f"Transforming data by selecting {len(self.selected_columns)} most important features."
        )
        try:

            X_selected = X[self.selected_columns].copy()
            if y is not None:
                if isinstance(y, list):
                    y = pd.Series(y)
                y_name = y.name if y.name is not None else "y"
                logger.debug(
                    f"Appending target variable '{y_name}' to the transformed data."
                )
                X_selected[y_name] = y.values
        except Exception as e:
            logger.error(f"Error in FeatureImportanceRegressionSelector transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X_selected

    def fit_transform(self, X: pd.DataFrame, y):
        """
        Fits and transforms the data by selecting the top k% most important features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.
        """

        logger.start_operation(
            f"Fitting and transforming data with top {self.k}% important features."
        )
        try:
            self.fit(X, y)
            X = self.transform(X, y)
        except Exception as e:
            logger.error(
                f"Error in FeatureImportanceRegressionSelector fit_transform: {e}"
            )
            raise e
        finally:
            logger.end_operation()
        return X

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "desc": f"Selects the top {self.k}% (rounded to whole number) of features most important according to Random Forest model for regression. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"k": self.k},
        }
