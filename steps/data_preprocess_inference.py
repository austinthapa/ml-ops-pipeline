import logging
import joblib
import pandas as pd

from pandas import DataFrame, Series
from zenml import step
from typing import Annotated, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s- %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)



@step(enable_cache=False)
def preprocess_input_inference(
    X: DataFrame,
    preprocessor_path: str = "preprocessor"
) -> Annotated[DataFrame, "X_processed"]:
    """
    Preprocess the input for before inference.
    """
    logger.info(f"Starting preprocessing input for inference...")
    
    try:
        
        # 1 - Validate and Prepare Data.
        X = X.drop("Outcome", axis = 1, errors="ignore").copy()
        
        # 2 - Load Standard Scaler and  scale the numeric columns.
        scaler = joblib.load(f"{preprocessor_path}/scaler.joblib")
        num_cols = ['Age', 'Cholesterol', 'BloodPressure', 'HeartRate', 'BMI','PhysicalActivity', 'AlcoholConsumption', 
                            'StressLevel', 'Income', 'MaxHeartRate', 'ST_Depression', 'NumberOfMajorVessels']
        X[num_cols] = scaler.transform(X[num_cols])
        
        # 3 - Load Ordinal Encoder and preprocess the required columns.
        ordinal_encoder = joblib.load(f"{preprocessor_path}/ordinal_encoder.joblib")
        ordinal_cols = ["Diet", "EducationLevel"]
        X[ordinal_cols] = ordinal_encoder.transform(X[ordinal_cols])
        
        # 4 - Load OneHotEncoder and encode the onehot columns.
        onehot_encoder = joblib.load(f"{preprocessor_path}/onehot_encoder.joblib")
        onehot_cols = ['Ethnicity','ChestPainType', 'ECGResults', 'Slope', 'Thalassemia', 'Residence', 'EmploymentStatus', 'MaritalStatus']
        encoded_df = DataFrame(onehot_encoder.transform(X[onehot_cols]), columns = onehot_encoder.get_feature_names_out(onehot_cols), index=X.index)
        X = pd.concat([X.drop(onehot_cols, axis = 1), encoded_df])

        # 5 - Map the binary categories to numbers.
        X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0})
        X["Medication"] = X["Medication"].map({"Yes": 1, "No": 0})
        X["ExerciseInducedAngina"] = X["ExerciseInducedAngina"].map({"Yes": 1, "No": 0})
        
        logger.info(f"Preprocessing for input complete...")
        return X
    except FileNotFoundError as e:
        logger.error(f"File Not Found at given location. {e}")
        raise
    except Exception as e:
        logger.error(f"An Unexpected error occured during preprocessing before inference: {e}", exc_info=True)
        raise