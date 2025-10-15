import logging
import pandas as pd

from pandas import DataFrame
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

logging.basicConfig(level=logging.INFO)

class Strategy(ABC):
    """
    Strategy Pattern
    """
    @abstractmethod
    def execute(self, df):
        pass
    
class BinarizeStrategy(Strategy):
    """
    Binarize the columns in the DataFrame.
    This will convert the two categorical columns into 0 & 1.
    """
    def execute(self, df):
        try:
            binary_cols = ["Gender", "Medication", "ExerciseInducedAngina", "Outcome"]
            missing_cols = [col for col in binary_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
            df["Medication"] = df["Medication"].map({"Yes": 1, "No": 0})
            df["ExerciseInducedAngina"] = df["ExerciseInducedAngina"].map({"Yes": 1, "No": 0})
            df["Outcome"] = df["Outcome"].map({"Heart Attack": 1, "No Heart Attack": 0})
            logging.info(f"Binarization completed...")
            return df
        except ValueError as ve:
            logging.error(f"Value Error: {ve}")
        except Exception as e:
            logging.error(f"Unexpected error occured: {e}")
            raise e
    
class OneHotEncodeStrategy(Strategy):
    """
    One Hot Encode categorical columns.
    This will convert categorical columns into separate binary columns for each category.
    """
    def execute(self, df):
        try:
            onehot_cols = ['Ethnicity','ChestPainType', 'ECGResults', 'Slope', 'Thalassemia', 'Residence', 'EmploymentStatus', 'MaritalStatus']
            onehot_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            encoded_array = onehot_encoder.fit_transform(df[onehot_cols])
            encoded_cols = onehot_encoder.get_feature_names_out(onehot_cols)
            encoded_df = DataFrame(encoded_array, columns=encoded_cols, index = df.index)
            
            df = df.drop(onehot_cols, axis = 1).join(encoded_df)
            logging.info("Onehot encoded successfully.")
            return df
        except Exception as e:
            logging.error(f"Unexpected Error: {e}")
            raise
 
class OrdinalEncodeStrategy(Strategy):
    """
    Ordinal Categorize the columns to preserve the natural order
    """
    def execute(self, df):
        try:
            ordinal_cols = ["Diet", "EducationLevel"]
            ordinal_encoder = OrdinalEncoder(
                categories=[["Healthy", "Moderate", "Unhealthy"],["High School", "College", "Postgraduate"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            df[ordinal_cols] = ordinal_encoder.fit_transform(df[ordinal_cols])
            
            logging.info(f"Ordinal Encoded Successfully...")
            return df
        except Exception as e:
            logging.error(f"Unexpected Error occured: {e}")
            raise
        
class StandardizeColumnsStrategy(Strategy):
    """
    Standardize the numerical columns using StandardScaler.
    This will transform features to have mean of 0 and variance of 1.
    """
    def execute(self, df):
        try:
            num_cols = ['Age', 'Cholesterol', 'BloodPressure', 'HeartRate', 'BMI','PhysicalActivity', 'AlcoholConsumption', 
                        'StressLevel', 'Income', 'MaxHeartRate', 'ST_Depression', 'NumberOfMajorVessels']
            missing_cols = [col for col in num_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            logging.info("Standardized Numerical Columns Successfully...")
            return df
        except Exception as e:
            logging.error(f"Unexpected error occured: {e}")
            raise
    
class DataPreprocess:
    def __init__(self, df: DataFrame, strategy: Strategy):
        self.df = df
        self.strategy = strategy
    
    def execute(self):
        return self.strategy.execute(self.df)        