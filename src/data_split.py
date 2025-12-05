import logging

from typing import Tuple
from pandas import DataFrame, Series
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

class Strategy(ABC):
    """
    Abstract Base Class for Strategy pattern.
    """
    @abstractmethod
    def execute(self, df: DataFrame) -> Tuple[DataFrame, DataFrame, Series, Series]:
        pass
    
class TrainTestSplitStrategy(Strategy):
    """
    Strategy for performing a standard train-test split on a dataset.

    Requirements:
        The DataFrame must contain a column named `'Outcome'` which is used
        as the target variable.

    Parameters:
        df(pandas.DataFrame) :The input dataset.

    Returns:
        (X_train, X_test, y_train, y_test): The resulting training and testing feature/label splits.

    Raises:
        ValueError: If the required `'Outcome'` column is not present.
        Exception: For unexpected errors (logged and re-raised).
    """
    def execute(self, df: DataFrame, test_size: float = 0.2, random_state:int = 42) -> Tuple[DataFrame, DataFrame, Series, Series]:
        try:
            if "Outcome" not in df.columns:
                raise ValueError("Expected column 'Outcome' not found in dataframe")
            X = df.drop(columns=["Outcome"])
            y = df["Outcome"].map({
                "Heart Attack": 1,
                "No Heart Attack": 0
            })
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=random_state)
            logging.info("Train & Test split successfully created:\n"
                         f"Train shape: {X_train.shape}\n"
                         f"Test shape: {X_test.shape}\n")
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Unexpected error occured during train test split: {e}")
            raise
        
class StratifiedSplitStrategy(Strategy):
    """
    Strategy for performing a stratified train-test split.

    Requirements:
        The DataFrame must contain a column `'Outcome'` that acts as the target.

    Parameters:
        df(pandas.DataFrame): The input dataset.

    Returns:
        (X_train, X_test, y_train, y_test): The stratified training and testing splits.

    Raises:
        ValueError: If `'Outcome'` column is missing.
        Exception: For unexpected errors (logged and re-raised).
    """
    def execute(self, df: DataFrame) -> Tuple[DataFrame, DataFrame, Series, Series]:
        try:
            if "Outcome" not in df.columns:
                raise ValueError("Expected column 'Outcome' not found in dataframe")
            X = df.drop(columns=["Outcome"])
            y = df["Outcome"].map({
                "Heart Attack": 1,
                "No Heart Attack": 0
            })
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, shuffle=True, random_state=42)
            logging.info("Stratified Train & Test Split Complete")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Unexpected error occured during stratified train test split: {e}")
            raise
        
class DataSplit:
    """
    Context Class for Strategy Pattern.
    """
    def __init__(self, df: DataFrame, strategy: Strategy):
        self.df = df
        self.strategy = strategy
    
    def execute(self) -> Tuple[DataFrame, DataFrame, Series, Series]:
        return self.strategy.execute(self.df)