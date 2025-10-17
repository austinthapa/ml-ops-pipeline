import logging

from pandas import DataFrame, Series
from zenml import step
from typing import Annotated
from sklearn.base import ClassifierMixin
from src.model_training import LogisticRegressionModel, DecisionTreeModel, SVCModel, GradientBoostingModel, RandomForestModel, NeuralNetworkModel, ModelTrainer

logging.basicConfig(level=logging.INFO)

# Dictionary to map model names to model classes
MODELS = {
    "log_reg_clf": LogisticRegressionModel,
    "decision_tree_clf": DecisionTreeModel,
    "svm_clf": SVCModel,
    "random_forest_clf": RandomForestModel,
    "gradient_boost_clf": GradientBoostingModel,
    "neural_net_clf": NeuralNetworkModel
}

@step
def train_model(
    model_name: str,
    X_train: DataFrame, 
    y_train: Series
) -> Annotated[ClassifierMixin, "trained_model"]:
    """
    Trains the model.
    
    Args:
        model_name (str): Name of the model
        X_train (DataFrame): Train datasets
        y_train (Series): Train Labels
    
    Returns:
        Trained Classifier Model
    
    Raises:
        ValueError: if model name is not supported.    
    """
    if model_name not in MODELS:
        logging.error(f"Unsupported model name: {model_name}")
        raise ValueError(f"Model '{model_name}' not supported. Please select from {MODELS.keys()}")
    
    logging.info(f"Training model: {model_name}")
    try:
        model_class = MODELS[model_name]
        model = ModelTrainer(model_class()).train(X_train, y_train)
        
        logging.info(f"Model: {model_name} successfully trained.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {model_name}")
        raise