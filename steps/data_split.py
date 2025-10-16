import logging
import pandas as pd

from pandas import DataFrame, Series
from zenml import step
from typing import Annotated, Tuple
from src.data_split import DataSplit, TrainTestSplitStrategy, StratifiedSplitStrategy

@step(enable_cache=True)
def split_data(df: DataFrame) -> Tuple[
    Annotated[DataFrame, "X_train"],
    Annotated[DataFrame, "X_test"],
    Annotated[Series, "y_train"],
    Annotated[Series, "y_test"]
]:
    """
    Splits the dataframe into train and test split
    
    Args:
        df (DataFrame): preprocessed dataframe
    
    Returns:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    X_train, X_test, y_train, y_test = DataSplit(df, TrainTestSplitStrategy()).execute()
    return X_train, X_test, y_train, y_test 
    
    
    
