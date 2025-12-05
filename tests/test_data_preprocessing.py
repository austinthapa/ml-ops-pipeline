import pytest
import pandas as pd
from unittest.mock import patch
from data_preprocessing import (
    load_config,
    BinarizeStrategy,
    OneHotEncodeStrategy, 
    OrdinalEncodeStrategy,
    StandardizeColumnsStrategy,
    DataPreprocess
)

@pytest.fixture
def mock_config_binarize():
    return {
        "features": {
            "binary": ["column1"]
        },
        "binary_mappings": {
            "column1": {"A": 1, "B": 0}
        }
    }

@pytest.fixture
def mock_config_onehot():
    return {
        "features": {
            "onehot": ["column1"]
        }
    }

@pytest.fixture
def mock_config_ordinal():
    return {
        "features": {
            "ordinal": ["column1"]
        },
        "ordinal_categories": {
            "column1": ["Low", "Medium", "High"]
        }
    }

@pytest.fixture
def mock_config_standardize():
    return {
        "features": {
            "numeric": ["column1", "column2"]
        }
    }
    
@pytest.fixture
def df_binarize():
    return pd.DataFrame({
        'column1': ['A', 'B', 'A', 'B'],
        'column2': [10, 20, 30, 40]
    })
    
@pytest.fixture
def df_onehot():
    return pd.DataFrame({
        'column1': ['A', 'B', 'A', 'C'],
        'column2': [10, 20, 30, 40]
    })
    
@pytest.fixture
def df_ordinal():
    return pd.DataFrame({
        'column1': ['Medium', 'Low', 'High', 'Low'],
        'column2': [10, 20, 30, 40]
    })
    
@pytest.fixture
def df_standardize():
    return pd.DataFrame({
        'column1': [10, 20, 30, 40],
        'column2': [100, 200, 300, 400]
    })
    
def test_load_config_valid():
    """
    This test ensures that the load_config function properly loads a valid YAML configuration file.
    """
    config = load_config("config/columns.yaml")
    assert isinstance(config, dict)
    assert "features" in config

def test_load_config_file_not_found():
    """
    This test ensures that the load_config function raises a FileNotFoundError
    when attempting to load a non-existent config file.
    """
    with pytest.raises(FileNotFoundError):
        load_config("config/non_existent.yaml")

def test_binarize_strategy(df_binarize, mock_config_binarize):
    """
    This test verifies that the BinarizeStrategy correctly binarizes a column according to a specified mapping.
    It checks that values in the 'column1' are transformed correctly from 'A' to 1 and 'B' to 0.
    """
    with patch('data_preprocessing.CONFIG', mock_config_binarize):
        strategy = BinarizeStrategy()
        processed_df = strategy.execute(df_binarize)

    assert 'column1' in processed_df.columns
    assert processed_df['column1'].iloc[0] == 1
    assert processed_df['column1'].iloc[1] == 0
    assert processed_df['column1'].iloc[2] == 1
    assert processed_df['column1'].iloc[3] == 0

def test_onehot_encode_strategy(df_onehot, mock_config_onehot):
    """
    This test checks that the OneHotEncodeStrategy correctly adds one-hot encoded columns
    for categorical variables. It verifies that the number of columns increases and contains the expected binary columns.
    """
    with patch('data_preprocessing.CONFIG', mock_config_onehot):
        strategy = OneHotEncodeStrategy()
        processed_df = strategy.execute(df_onehot)

    assert 'column1_B' in processed_df.columns
    assert 'column1_C' in processed_df.columns
    assert processed_df.shape[1] == 3

def test_ordinal_encode_strategy(df_ordinal, mock_config_ordinal):
    """
    This test ensures that the OrdinalEncodeStrategy correctly encodes categorical values in 'column1'
    based on predefined categories: Low, Medium, and High.
    """
    with patch('data_preprocessing.CONFIG', mock_config_ordinal):
        strategy = OrdinalEncodeStrategy()
        processed_df = strategy.execute(df_ordinal)

    assert 'column1' in processed_df.columns
    assert processed_df['column1'].iloc[0] == 1  # Medium
    assert processed_df['column1'].iloc[1] == 0  # Low
    assert processed_df['column1'].iloc[2] == 2  # High

def test_standardize_strategy(df_standardize, mock_config_standardize):
    """
    This test checks if the StandardizeColumnsStrategy standardizes numeric columns correctly,
    ensuring that the mean of 'column1' and 'column2' after processing is approximately 0.
    """
    with patch('data_preprocessing.CONFIG', mock_config_standardize):
        strategy = StandardizeColumnsStrategy()
        processed_df = strategy.execute(df_standardize)

    assert 'column1' in processed_df.columns
    assert 'column2' in processed_df.columns
    assert processed_df['column1'].mean() == pytest.approx(0, abs=1e-1)
    assert processed_df['column2'].mean() == pytest.approx(0, abs=1e-1)

def test_data_preprocess_binarize(df_binarize, mock_config_binarize):
    """
    This test ensures that the DataPreprocess class works as expected when using the BinarizeStrategy.
    It verifies that the binarization of 'column1' happens correctly, transforming values 'A' to 1 and 'B' to 0.
    """
    with patch('data_preprocessing.CONFIG', mock_config_binarize):
        strategy = BinarizeStrategy()
        processor = DataPreprocess(df_binarize, strategy)
        processed_df = processor.execute()

    assert 'column1' in processed_df.columns
    assert processed_df['column1'].iloc[0] == 1
    assert processed_df['column1'].iloc[1] == 0
    assert processed_df['column1'].iloc[2] == 1
    assert processed_df['column1'].iloc[3] == 0

def test_data_preprocess_onehot(df_onehot, mock_config_onehot):
    """
    This test verifies that the DataPreprocess class correctly applies the OneHotEncodeStrategy.
    It checks that the 'column1' is one-hot encoded into separate binary columns for each category.
    """
    with patch('data_preprocessing.CONFIG', mock_config_onehot):
        strategy = OneHotEncodeStrategy()
        processor = DataPreprocess(df_onehot, strategy)
        processed_df = processor.execute()

    assert 'column1_B' in processed_df.columns
    assert 'column1_C' in processed_df.columns
    assert processed_df.shape[1] == 3
    
def test_data_preprocess_ordinal(df_ordinal, mock_config_ordinal):
    """
    This test checks that the DataPreprocess class applies the OrdinalEncodeStrategy correctly.
    It verifies that 'column1' is encoded into the ordinal values for the categories Low, Medium, and High.
    """
    with patch('data_preprocessing.CONFIG', mock_config_ordinal):
        strategy = OrdinalEncodeStrategy()
        processor = DataPreprocess(df_ordinal, strategy)
        processed_df = processor.execute()

    assert 'column1' in processed_df.columns
    assert processed_df['column1'].iloc[0] == 1  # Medium
    assert processed_df['column1'].iloc[1] == 0  # Low
    assert processed_df['column1'].iloc[2] == 2  # High

def test_data_preprocess_standardize(df_standardize, mock_config_standardize):
    """
    This test checks that the DataPreprocess class applies the StandardizeColumnsStrategy correctly.
    It verifies that 'column1' is scaled with mean of 0 and standard deviation of 1.
    """
    with patch('data_preprocessing.CONFIG', mock_config_standardize):
        strategy = StandardizeColumnsStrategy()
        processor = DataPreprocess(df_standardize, strategy)
        processed_df = processor.execute()

    assert 'column1' in processed_df.columns
    assert 'column2' in processed_df.columns
    assert processed_df['column1'].mean() == pytest.approx(0, abs=1e-1)
    assert processed_df['column2'].mean() == pytest.approx(0, abs=1e-1)
