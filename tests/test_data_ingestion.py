import pytest
import pandas as pd
from data_split import (
    TrainTestSplitStrategy, StratifiedSplitStrategy, DataSplit
)

@pytest.fixture
def mock_dataframe():
    """
    This fixture creates a mock DataFrame that is used in all tests. The DataFrame contains features and an 'Outcome' column required by both train-test split strategies.
    """
    data = {
        'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'Outcome': ['Heart Attack', 'No Heart Attack', 'Heart Attack', 'No Heart Attack', 'Heart Attack', 
                    'No Heart Attack', 'Heart Attack', 'No Heart Attack', 'Heart Attack', 'No Heart Attack']
    }
    return pd.DataFrame(data)

def test_train_test_split(mock_dataframe):
    """
    This test verifies that the `TrainTestSplitStrategy` works correctly.
    """
    strategy = TrainTestSplitStrategy()
    X_train, X_test, y_train, y_test = strategy.execute(mock_dataframe)

    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2

    assert 'Feature1' in X_train.columns
    assert 'Outcome' not in X_train.columns
    assert 'Feature2' in X_train.columns

    assert 'Outcome' not in X_train.columns
    assert 'Outcome' in y_train.name

def test_train_test_split_missing_outcome(mock_dataframe):
    """
    This test checks that the `TrainTestSplitStrategy` raises a ValueError when the 'Outcome' column is missing.
    """
    df_missing_outcome = mock_dataframe.drop(columns=['Outcome'])

    strategy = TrainTestSplitStrategy()
    with pytest.raises(ValueError, match="Expected column 'Outcome' not found in dataframe"):
        strategy.execute(df_missing_outcome)

def test_stratified_split(mock_dataframe):
    """
    This test verifies that the `StratifiedSplitStrategy` works correctly.
    """
    strategy = StratifiedSplitStrategy()
    X_train, X_test, y_train, y_test = strategy.execute(mock_dataframe)
    
    assert y_train.value_counts().get(1, 0) == 4
    assert y_train.value_counts().get(0, 0) == 4
    assert y_test.value_counts().get(1, 0) == 1
    assert y_test.value_counts().get(0, 0) == 1

def test_stratified_split_missing_outcome(mock_dataframe):
    """
    This test checks that the `StratifiedSplitStrategy` raises a ValueError when the 'Outcome' column is missing.
    It simulates the scenario where the DataFrame does not contain the required 'Outcome' column.
    """
    df_missing_outcome = mock_dataframe.drop(columns=['Outcome'])

    strategy = StratifiedSplitStrategy()
    with pytest.raises(ValueError, match="Expected column 'Outcome' not found in dataframe"):
        strategy.execute(df_missing_outcome)

def test_data_split(mock_dataframe):
    """
    This test ensures that the `DataSplit` context class correctly delegates the execution 
    to the strategy (either `TrainTestSplitStrategy` or `StratifiedSplitStrategy`).
    """
    # Test with TrainTestSplitStrategy
    strategy = TrainTestSplitStrategy()
    data_split = DataSplit(mock_dataframe, strategy)
    X_train, X_test, y_train, y_test = data_split.execute()
    
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2

    # Test with StratifiedSplitStrategy
    strategy = StratifiedSplitStrategy()
    data_split = DataSplit(mock_dataframe, strategy)
    X_train, X_test, y_train, y_test = data_split.execute()

    assert y_train.value_counts().get(1, 0) == 4
    assert y_test.value_counts().get(1, 0) == 1
