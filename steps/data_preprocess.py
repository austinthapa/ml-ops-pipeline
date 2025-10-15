import logging
import pandas as pd

from zenml import step
from pandas import DataFrame
from typing import Annotated
from src.data_cleaning import StandardizeColumnsStrategy, BinarizeStrategy, OneHotEncodeStrategy, OrdinalEncodeStrategy, DataPreprocess
logging.basicConfig(level=logging.INFO)

@step
def preprocess_data(df: DataFrame) -> Annotated[DataFrame, "cleaned_df"]:
    """
    Cleans the dataframe, onehot encode, ordinal encode, binarize, and scale the columns
    
    Args:
        DataFrame: Raw DataFrame
        
    Returns:
        DataFrame: Encoded DataFrame ready to train ML model.
    """
    try:
        df = DataPreprocess(df, StandardizeColumnsStrategy()).execute()
        df = DataPreprocess(df, BinarizeStrategy()).execute()
        df = DataPreprocess(df, OrdinalEncodeStrategy()).execute()
        df = DataPreprocess(df, OneHotEncodeStrategy()).execute()
        return df
    except Exception as e:
        logging.error("Error during data preprocessing")
        raise
    