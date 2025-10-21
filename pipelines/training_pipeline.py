import logging

from zenml import pipeline
from steps.data_ingestion import ingest_data
from steps.data_preprocess import preprocess_data
from steps.data_split import split_data
from steps.model_train import train_model
from steps.model_evaluation import evaluate_model

logging.basicConfig(level=logging.INFO)

@pipeline(enable_cache=True)
def training_pipeline(
    data_path: str,
    data_format: str,
    model_name: str
):
    # Step 1 -- Ingest Data
    ingested_df = ingest_data(data_path, data_format)
    
    # Step 2 -- Preprocess Data
    preprocess_df = preprocess_data(ingested_df)
    
    # Step 3 -- Split Data
    X_train, X_test, y_train, y_test = split_data(preprocess_df)
    
    # Step 4 -- Train Model
    model = train_model(model_name, X_train, y_train)
    
    # Step 5 -- Evaluate Model
    test_score = evaluate_model(model, X_test, y_test)
    logging.info(f"Test Score: {test_score}")
    return test_score