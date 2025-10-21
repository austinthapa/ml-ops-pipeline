import logging

from zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step

from steps.data_ingestion import ingest_data
from steps.data_preprocess import preprocess_data
from steps.data_split import split_data
from steps.model_train import train_model
from steps.model_evaluation import evaluate_model
from config.pipeline_config import PIPELINE_CONFIG

logging.basicConfig(level=logging.INFO)

docker_settings = DockerSettings(required_integrations=[MLFLOW])

MIN_ACCURACY_THRESHOLD = PIPELINE_CONFIG["min_accuracy_threshold"]

@step(enable_cache=True)
def deployment_trigger(
    accuracy: float,
    min_accuracy: float = MIN_ACCURACY_THRESHOLD
) -> bool:
    """
    Decide whether to deploy the model or not based on accuracy threshold.
    """
    if not 0 <= accuracy <= 1:
        raise ValueError(f"Accuracy must be between 0 and 1, got {accuracy}")
    deployment_decision = accuracy > min_accuracy
    logging.info(f"Deployment decision: {deployment_decision}"
                 f"(Accuracy= {accuracy:.2f}, Min Accuracy Required: {min_accuracy:.2f})")
    return accuracy > min_accuracy


@pipeline(
    enable_cache=True,
    docker_settings=docker_settings,
    name="continuous_deployment_pipeline",
    description="End-to-end ML pipeline with continuous deployment to MLflow"
)
def continuous_deployment_pipeline(
    data_path: str,
    data_format: str,
    model_name: str
):
    """
    End-to-end continuous deployment pipeline.
    """
    # Step 1 -- Ingest Data
    df = ingest_data(data_path, data_format)
    
    # Step 2 -- Preprocess Data
    df = preprocess_data(df)
    
    #Step 3 -- Split Data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Step 4 -- Train Model
    model = train_model(model_name, X_train, y_train)
    
    # Step 5 -- Evaluate Model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 6 -- Deployment Decision
    deployment_decision = deployment_trigger(accuracy)
    
    # Step 7 -- Deploy the model if it meets required threshold
    mlflow_model_deployer_step(
        model = model,
        deployment_decision = deployment_decision
    )