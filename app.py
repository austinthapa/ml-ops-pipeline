import joblib
import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)
logger = logging.getLogger(__name__)

class PatientData(BaseModel):
    age: float = Field(..., ge=0, le=120, alias="Age")
    gender: str = Field(..., alias="Gender")
    cholesterol: float = Field(..., ge=0, alias="Cholesterol")
    blood_pressure: float = Field(..., ge=0, alias="BloodPressure")
    heart_rate: float = Field(..., ge=0, alias="HeartRate")
    bmi: float = Field(..., ge=0, alias="BMI")
    smoker: int = Field(..., alias="Smoker")
    diabetes: int = Field(..., alias="Diabetes")
    hypertension: int = Field(..., alias="Hypertension")
    family_history: int = Field(..., alias="FamilyHistory")
    physical_activity: float = Field(..., alias="PhysicalActivity")
    alcohol_consumption: float = Field(..., alias="AlcoholConsumption")
    diet: str = Field(..., alias="Diet")
    stress_level: float = Field(..., alias="StressLevel")
    ethnicity: str = Field(..., alias="Ethnicity")
    income: float = Field(..., ge=0, alias="Income")
    education_level: str = Field(..., alias="EducationLevel")
    medication: str = Field(..., alias="Medication")
    chest_pain_type: str = Field(..., alias="ChestPainType")
    ecg_results: str = Field(..., alias="ECGResults")
    max_heart_rate: float = Field(..., ge=0, alias="MaxHeartRate")
    st_depression: float = Field(..., ge=0, alias="ST_Depression")
    exercise_induced_angina: str = Field(..., alias="ExerciseInducedAngina")
    slope: str = Field(..., alias="Slope")
    number_of_major_vessels: float = Field(..., ge=0, le=4, alias="NumberOfMajorVessels")
    thalassemia: str = Field(..., alias="Thalassemia")
    previous_heart_attack: int = Field(..., alias="PreviousHeartAttack")
    stroke_history: int = Field(..., alias="StrokeHistory")
    residence: str = Field(..., alias="Residence")
    employment_status: str = Field(..., alias="EmploymentStatus")
    marital_status: str = Field(..., alias="MaritalStatus")

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True
        schema_extra = {
            "example": {
                "age": 58.0,
                "gender": "Male",
                "cholesterol": 234.0,
                "blood_pressure": 150.0,
                "heart_rate": 160.0,
                "bmi": 25.5,
                "smoker": 0,
                "diabetes": 0,
                "hypertension": 1,
                "family_history": 1,
                "physical_activity": 5.0,
                "alcohol_consumption": 0.0,
                "diet": "Moderate",
                "stress_level": 3.0,
                "ethnicity": "Caucasian",
                "income": 60000.0,
                "education_level": "College",
                "medication": "Yes",
                "chest_pain_type": "Atypical",
                "ecg_results": "Normal",
                "max_heart_rate": 180.0,
                "st_depression": 1.5,
                "exercise_induced_angina": "No",
                "slope": "Upsloping",
                "number_of_major_vessels": 2.0,
                "thalassemia": "Normal",
                "previous_heart_attack": 0,
                "stroke_history": 0,
                "residence": "Urban",
                "employment_status": "Employed",
                "marital_status": "Married"
            }
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        artifact_path = Path("artifacts")
        
        with open("config/columns.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        logger.info("Column configuration successfully loaded...")
        
         # --- Load the scaler ---
        scaler = joblib.load(artifact_path / "scaler.joblib")
        logger.info("Scaler loaded successfully...")
        
        # --- Load the One Hot Encoder ---
        onehot_encoder = joblib.load(artifact_path / "onehot_encoder.joblib")
        logger.info("OneHot encoder loaded successfully...")
        
        # --- Load the Ordinal Encoder ---
        ordinal_encoder = joblib.load(artifact_path / "ordinal_encoder.joblib")
        logger.info("Ordinal encoder loaded successfully...")
        
        # --- Load the model ---
        model = joblib.load(artifact_path / "model.joblib")
        logger.info("Model loaded successfully...")
        
        app.state.config = config
        app.state.scaler = scaler
        app.state.onehot_encoder = onehot_encoder
        app.state.ordinal_encoder = ordinal_encoder
        app.state.model = model
        
        yield
        # Cleanup (if needed)
        logger.info("Shutting down service...")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading artifacts: {e}", exc_info=True)
        raise

app = FastAPI(
    title="Heart Attack Prediction Service",
    description="A service for predicting heart attack risk based on clinical data.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def home():
    return {
        "message": "Heart Attack Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }
    
@app.get("/health")
def health_check():
    return {
        "Status": "Okay",
        "Service": "Online"
    }

def preprocess_input(
    request: Request,
    data: PatientData
) -> np.ndarray:
    """
    Applies the necessary transformations to the input data.
    """
    input_dict = data.model_dump(by_alias=True)
    df = pd.DataFrame(input_dict, index=[0])

    config = request.app.state.config

    # --- Scale numeric columns ---
    logger.info(f"Scaling numeric features started...")
    scaler = request.app.state.scaler
    num_cols = config["features"]["numeric"]
    logger.info("Numeric columns: \n"
                f"{num_cols}")
    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])
    logger.info(f"Scaling numeric features completed...")
    
    # --- Onehot encode ---
    logger.info(f"Onehot encoding features started...")
    onehot_encoder = request.app.state.onehot_encoder
    onehot_cols = config["features"]["onehot"]
    if onehot_cols:
        onehot_encoded = onehot_encoder.transform(df[onehot_cols])
        feature_names = onehot_encoder.get_feature_names_out(onehot_cols)
        onehot_df = pd.DataFrame(onehot_encoded, columns=feature_names, index=df.index)
        df = df.drop(columns=onehot_cols)
        df = pd.concat([df, onehot_df], axis=1)
    logger.info(f"Onehot encoding features completed...")
    
    # --- Ordinal encode ---
    logger.info(f"Ordinal encoding features started...")
    ordinal_encoder = request.app.state.ordinal_encoder
    ordinal_cols = config["features"]["ordinal"]
    if ordinal_cols:
        df[ordinal_cols] = ordinal_encoder.transform(df[ordinal_cols])
    logger.info(f"Ordinal encoding features started...")
    
    # --- Binary map ---
    logger.info(f"Binary encoding features started...")
    binary_mappings = config["binary_mappings"]
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            if df[col].isna().any():
                raise ValueError(f"Invalid value for {col}. Expected one of {list(mapping.keys())}")
    logger.info(f"Binary encoding features completed...")
    
    model = request.app.state.model
    feature_order = model.feature_names_in_

    return df[feature_order]

@app.post("/predict")
async def predict_heart_attack(
    request: Request,
    data: PatientData
):
    """
    Predict heart attack risk based on patient data.
    """
    try:
        model = request.app.state.model
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Check server logs."
            )

        processed_data = preprocess_input(request, data)

        prediction = model.predict(processed_data)

        return {
            "prediction": int(prediction[0]),
            "risk_level": "High Risk" if prediction[0] == 1 else "Low Risk"
        }

    except ValueError as e:
        logger.error(f"Validation Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


