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
def continuous_deployment_pipeline():
    """
    End-to-end continuous deployment pipeline.
    
    Steps:
        1. Ingest data
        2. Preprocess data
        3. Split data
        4. Train model
        5. Evaluate model
        6. Deploy model to MLflow if accuracy threshold is met
    """
    try:
        logging.info("Starting continuous deployment pipeline..."
                    "Step 1: Ingesting data...")
        df = ingest_data()
        
        logging.info("Step 2: Preprocessing data...")
        df = preprocess_data(df)
        
        logging.info("Step 3: Splitting data...")
        X_train, X_test, y_train, y_test = split_data(df)
        
        # Model pipeline
        logging.info("Step 4: Training model...")
        model = train_model(X_train, y_train)
        
        logging.info("Step 5: Evaluating model...")
        accuracy = evaluate_model(model, X_test, y_test)
        deployment_decision = deployment_trigger(accuracy)
        
        mlflow_model_deployer_step(
            model = model,
            deployment_decision = deployment_decision
        )
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise