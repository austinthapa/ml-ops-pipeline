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
    Split the dataset into train and test split
    """
    def execute(self, df: DataFrame) -> Tuple[DataFrame, DataFrame, Series, Series]:
        try:
            if "Outcome" not in df.columns:
                raise ValueError("Expected column 'Outcome' not found in dataframe")
            X = df.drop(columns=["Outcome"])
            y = df["Outcome"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)
            logging.info("Train & Test split successfully created")    
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Unexpected error occured during train test split: {e}")
            raise e
class StratifiedSplitStrategy(Strategy):
    """
    Split the dataset into train and test split based on the distribution
    """
    def execute(self, df: DataFrame) -> Tuple[DataFrame, DataFrame, Series, Series]:
        try:
            if "Outcome" not in df.columns:
                raise ValueError("Expected column 'Outcome' not found in dataframe")
            X = df.drop(columns=["Outcome"])
            y = df["Outcome"]
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