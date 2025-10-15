import logging
import pandas as pd

from zenml import step
from pandas import DataFrame
from typing import Annotated
from pathlib import Path

logging.basicConfig(level=logging.INFO)

@step
def ingest_data(
    data_path: str,
    data_format: str
) -> Annotated[DataFrame, "ingested_df"]:
    """
    Ingest the data from specified location and specified format.

    Args:
        data_path (str): Location where data resides.
        data_format (str): Format in which data exists.
    
    Returns:
        DataFrame: Ingested data in DataFrame
    
    Raises:
        ValueError: If the file format in not supported.
        FileNotFoundError: If the file does not exists at given location.
        Exception: Other unrelated exception that occurs.
    """
    logging.info("Staring data ingestion...")
    logging.info(f"Data Format Specified: {data_format}")

    data_path_obj= Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"File Not Found at given location: {data_path}")
    
    try:
        if data_format == "csv":
            df = pd.read_csv(data_path_obj)
        elif data_format == "parquet":
            df = pd.read_parquet(data_path_obj)
        elif data_format == "excel":
            df = pd.read_excel(data_path_obj)
        else:
            raise ValueError(f"Unsupported Data File Format: {data_format}. \n Supported Formats are: [csv, parquet, excel]")
        
        if df.empty:
            logging.error("Ingested dataframe is empty")
            raise ValueError("Empty dataframe ingested")
        logging.info(f"Data Ingestion Successful. Shape of df: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        logging.info(f"File is empty at given location: {data_path}")
        raise ValueError("Cannot ingest empty file")
    except Exception as e:
        logging.error(f"Unexpected error occured at: {e}")
        raise