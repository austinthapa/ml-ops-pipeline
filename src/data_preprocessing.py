import yaml
import logging
import joblib
import pandas as pd

from pandas import DataFrame
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)
logger = logging.getLogger(__name__)

# --- Helper function to load configuartion ---
def load_config(
    config_path = "config/columns.yaml"
):
    """
    Load and parse a YAML configuration file.

    Args:
        config_path (str): Defaults to "columns.yaml".
    Returns:
        dict: Parsed configuration data as a dictionary.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist at the given path.
        yaml.YAMLError: If the YAML file contains invalid syntax or cannot be parsed.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        logger.error(f"Config file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error while parsing YAML file: {e}")
        raise

CONFIG = load_config()

class Strategy(ABC):
    """
    Strategy Pattern
    """
    @abstractmethod
    def execute(self, df):
        pass
    
class BinarizeStrategy(Strategy):
    """
    Strategy for converting categorical columns into binary (0/1) format.

    This strategy reads:
        - `CONFIG["features"]["binary"]` for the list of binary columns.
        - `CONFIG["binary_mappings"]` for the value-to-binary mapping dicts.

    Args: 
        df (pandas.DataFrame): The input DataFrame containing the raw features.

    Returns:
        pandas.DataFrame: A DataFrame where the specified binary columns have been converted.
        
    Raises
        ValueError: If any required binary columns are missing from the DataFrame.
        Exception: For unexpected processing errors; re-raised after logging.
    """
    def execute(self, df):
        try:
            binary_cols = CONFIG["features"]["binary"]
            binary_mappings = CONFIG["binary_mappings"]
            missing_cols = [col for col in binary_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            for col, mapping in binary_mappings.items():
                df[col] = df[col].map(mapping)
                
            logging.info(f"Binarization completed...")
            return df
        except ValueError as ve:
            logging.error(f"Value Error: {ve}")
        except Exception as e:
            logging.error(f"Unexpected error occured: {e}")
            raise e
    
class OneHotEncodeStrategy(Strategy):
    """
    Strategy for one-hot encoding categorical columns.

    Args:
        df (pandas.DataFrame): Input DataFrame containing raw categorical features.

    Returns:
        pandas.DataFrame: A DataFrame where the original one-hot columns have been replaced by
        their encoded binary indicator columns.

    Raises:
        ValueError: If any expected one-hot columns are missing from the input DataFrame.
        Exception: For unexpected errors; the exception is logged and re-raised.
    """
    def execute(self, df):
        try:
            onehot_cols = CONFIG["features"]["onehot"]
            missing_cols = [col for col in onehot_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            onehot_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            encoded_array = onehot_encoder.fit_transform(df[onehot_cols])
            joblib.dump(onehot_encoder, "artifacts/onehot_encoder.joblib")
            encoded_cols = onehot_encoder.get_feature_names_out(onehot_cols)
            encoded_df = DataFrame(encoded_array, columns=encoded_cols, index = df.index)
            
            df = df.drop(onehot_cols, axis = 1).join(encoded_df)
            logging.info("Onehot encoded successfully.")
            return df
        except Exception as e:
            logging.error(f"Unexpected Error: {e}")
            raise
 
class OrdinalEncodeStrategy(Strategy):
    """
    Strategy for ordinal encoding of categorical features that have a
    meaningful, predefined order.

    Args:
        df (pandas.DataFrame): Input DataFrame containing raw categorical variables.

    Returns: 
        pandas.DataFrame: A DataFrame where each ordinal column has been replaced with
        its corresponding integer-encoded values according to the
        provided category order.

    Raises:
        ValueError: If a required ordinal column is missing or has no defined category order.
        Exception: Any unexpected errors during encoding; the exception is logged and re-raised.
    """
    def execute(self, df):
        try:
            ordinal_cols = CONFIG["features"]["ordinal"]
            ordinal_categories_map = CONFIG["ordinal_categories"]
            missing_cols = [col for col in ordinal_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required ordinal columns: {missing_cols}")

            categories_list = [ordinal_categories_map[col] for col in ordinal_cols if col in ordinal_categories_map]
            
            ordinal_encoder = OrdinalEncoder(
                categories=categories_list,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            df[ordinal_cols] = ordinal_encoder.fit_transform(df[ordinal_cols])
            joblib.dump(ordinal_encoder, "artifacts/ordinal_encoder.joblib")
            logging.info(f"Ordinal Encoded Successfully...")
            return df
        except Exception as e:
            logging.error(f"Unexpected Error occured: {e}")
            raise
        
class StandardizeColumnsStrategy(Strategy):
    """
    Strategy for standardizing numerical features using `StandardScaler`.

    Args:
        df( pandas.DataFrame): Input DataFrame containing numerical columns to be standardized.

    Returns:
        pandas.DataFrame: The DataFrame with standardized numerical features.

    Raises:
        ValueError: If one or more required numerical columns are missing.
        Exception: Any unexpected error during processing; logged and re-raised.
    """
    def execute(self, df):
        try:
            num_cols = CONFIG["features"]["numeric"]
            missing_cols = [col for col in num_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            joblib.dump(scaler, "artifacts/scaler.joblib")
            logging.info("Standardized Numerical Columns Successfully...")
            return df
        except Exception as e:
            logging.error(f"Unexpected error occured: {e}")
            raise
    
class DataPreprocess:
    def __init__(self, df: DataFrame, strategy: Strategy):
        self.df = df
        self.strategy = strategy
    
    def execute(self):
        return self.strategy.execute(self.df)           