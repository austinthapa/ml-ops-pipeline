import logging

from pandas import DataFrame, Series
from zenml import step
from typing import Annotated
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

@step
def evaluate_model(
    model: ClassifierMixin,
    X_test: DataFrame, 
    y_test: Series) -> Annotated[float, "test_accuracy"]:
    """
    Evaluate the model
    
    Args:
        model (ClassifierMixin): A trained scikit-learn estimator.
        X_test (DataFrame): Test datasets with shape (n_samples, n_features).
        y_test (Series): Test labels with shape (n_samples, ).
    
    Returns:
        float: Test accuracy score [0-1].
    """
    try:
        y_predict = model.predict(X_test)
        score = accuracy_score(y_test, y_predict)
        logging.info(f"Test accuracy: {score}")
        return score
    except Exception as e:
        logging.error(f"Unexpected error occured during model evaluation: {e}")
        raise