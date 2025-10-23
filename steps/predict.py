import logging

from pandas import DataFrame
from zenml import step
from typing import Annotated, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s- %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@step(enable_cache=False)
def predict(
    X: DataFrame,
    model: Any
) -> Annotated[Any, "prediction"]:
    """
    Make prediction on the inputs using loaded model.
    
    Args:
        X: Input data
        model: Loaded model from Model Registry
    
    Returns:
        Prediction
    
    Raises:
        ValueError: If the input is null
    """
    try:
        if X is None:
            raise ValueError("Input data is required to make a prediction.")
        logger.info("Making prediction on given input...")
        y_predict = model.predict(X)
        logger.info("Prediction complete.")
        return y_predict
    except Exception as e:
        logger.error(f"An Unexpected error occured during making prediction: {e}", exc_info=True)
        raise