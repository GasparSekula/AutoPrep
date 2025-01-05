from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Numerical(ABC):
    """Abstract interface to indicate numerical step"""

    pass


class Categorical(ABC):
    """Abstract interface to indicate categorical step"""

    pass


class Step(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract class to be overwritten for implementing custom
    preprocessing steps. If step is parametrizable, it should have
    defined "param_grid" of all possible values for each parameter.
    """

    @abstractmethod
    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.
        Keys are: name - transformer name, desc - short description, params - class parameters (if None then {}).
        """
        pass

    @abstractmethod
    def is_numerical(self) -> bool:
        """
        If this step is for numerical data or not.

        Returns:
            bool: True if the step is for numerical data, False otherwise.
        """
        pass


class RequiredStep(Step):
    """
    Required step that will be always considered in preprocessing.
    """

    pass


class NonRequiredStep(Step):
    """
    Non required step that will be only considered for preprocessing
    if class method is_applicable returns True.
    """

    pass


class FeatureImportanceSelector(ABC, NonRequiredStep):
    """
    Transformer to select k% (rounded to whole number) of features
    that are most important according to Random Forest model.

    Attributes:
        k (float): The percentage of top features to keep based on their importance.
        selected_columns (list): List of selected columns based on feature importance.
    """

    def __init__(self, k: float = 10.0):
        """
        Initializes the transformer with a specified model and percentage of top important features to keep.

        Args:
            k (float): The percentage of features to retain based on their importance.
        """
        if not (0 <= k <= 100):
            raise ValueError("k must be between 0 and 100.")
        self.k = k
        self.selected_columns = []

    def fit(
        self,
        X,  # noqa: F841
        y,  # noqa: F841
    ):
        """
        Identifies the top k% (rounded to whole value) of features most important according to the model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceSelector: The fitted transformer instance.
        """
        pass

    def transform(
        self,
        X: pd.DataFrame,  # noqa: F841
        y: pd.Series = None,  # noqa: F841
    ):
        """
        Selects the top k% of features most important according to the model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits and transforms the data by selecting the top k% most important features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame: The transformed data with selected features.
        """
        self.fit(X, y)
        return self.transform(X)

    def is_applicable(
        self,
        dt: pd.DataFrame,  # noqa: F841
    ):
        """
        Args:
            dt (pd.Series) - column that is considered to be preprocessed
                by that transformer.

        Returns:
            bool - True if it is possible to use this transofmation
                on passed data.
        """
        return True
