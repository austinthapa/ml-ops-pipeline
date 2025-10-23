import logging
import mlflow

from zenml import step
from sklearn.base import ClassifierMixin
from typing import Any, Annotated

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s'
)
logger = logging.getLogger(__name__)


@step(enable_cache=False)
def load_model(
    model_name: str,
    stage: str = "Production"
) -> Annotated[Any, "loaded_model"]:
    """
    Load the models from MLflow Model Registry
    
    Args:
        model_path: Path to the model.
        model_name: Name of the registered model.
    
    Returns:
        The Loaded MLflow Object model
        
    Raises:
        RunTimeError: If the model cannot be found.
    """
    try:
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading the model from model URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Successfully loaded the model...")
        return model
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"Unable to load the model: {model_name} at stage: {stage}", exc_info=True)
        raise RuntimeError from e
    except Exception as e:
        logger.error(f"An Unexpected Error occured during loading the model: {e}", exc_info=True)
        raise
    
    