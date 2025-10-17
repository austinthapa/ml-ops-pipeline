import logging
import mlflow

from abc import ABC, abstractmethod
from pandas import DataFrame, Series
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)

class Model(ABC):
    """
    Base class for ML models
    """
    def __init__(self, **params):
        params.setdefault("random_state", 42)
        self.params = params
        self.model = None
    
    @abstractmethod
    def get_estimator(self) -> ClassifierMixin:
        """
        Subclasses must return an untrained scikit-learn estimator.
        """
        pass
    
    def train(self, X: DataFrame, y:Series) -> ClassifierMixin:
        try:
            logging.info(f"Training {self.__class__.__name__} with params: {self.params}")
            # MLFlow Run
            with mlflow.start_run(run_name=self.__class__.__name__):
                # Log params
                mlflow.log_params(self.params)
                
                self.model = self.get_estimator()
                self.model.fit(X, y)
                
                y_predict = self.model.predict(X)
                train_acc = accuracy_score(y, y_predict)
                
                # Log training metrics
                mlflow.log_metric("train_accuracy", train_acc)
                
                # Log model
                mlflow.sklearn.log_model(self.model, artifact_path="model")
            
                logging.info(f"{self.__class__.__name__} training accuracy: {train_acc}")
                return self.model
        except Exception as e:
            logging.error(f"Error training {self.__class__.__name__}: {e}")
            raise
        
class LogisticRegressionModel(Model):
    """
    Logistic Regresssion Model
    """
    def get_estimator(self) -> ClassifierMixin:
        return LogisticRegression(**self.params)
    
class DecisionTreeModel(Model):
    """
    Decision Tree Classifier Model
    """
    def get_estimator(self) -> ClassifierMixin:
        return DecisionTreeClassifier(**self.params)
    
class SVCModel(Model):
    """
    Support Vector Classifier Model
    """
    def get_estimator(self) -> ClassifierMixin:
        return SVC(**self.params)

class GradientBoostingModel(Model):
    """
    Gradient Boosting Classifier Model
    """
    def get_estimator(self) -> ClassifierMixin:
        return GradientBoostingClassifier(**self.params)
    
class RandomForestModel(Model):
    """
    Random Forest Classifier Model
    """
    def get_estimator(self) -> ClassifierMixin:
        return RandomForestClassifier(**self.params)
    
class NeuralNetworkModel(Model):
    """
    Neural Network Model
    """
    def get_estimator(self) -> ClassifierMixin:
        return 
    
class ModelTrainer:
    """
    Context Class to train the model
    """
    def __init__(self, model: Model) -> None:
        self.model = model
    
    def train(self, X: DataFrame, y:Series) -> ClassifierMixin:
        return self.model.train(X, y)
