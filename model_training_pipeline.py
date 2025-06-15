import logging
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.data_processing import NullHandler, DataProcessor, DataSampler
from src.data_loading import DataLoader
import config
from data_processing_pipeline import build_data_processing_pipeline
from sklearn.base import ClassifierMixin
import pandas as pd
import os

logging.basicConfig(
    filename='app.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelTrainingPipeline:
    """Pipeline for training a machine learning model with data processing steps.
    This pipeline includes data normalisation, handling null values, and model training.
    Attributes:
        normalisation_columns (list): List of columns to normalise.
        normalisation_strategy (str): Normalisation strategy to use ('minmax' or 'std_scaler').
        model (ClassifierMixin): The machine learning model to train.
    """

    def __init__(self, normalisation_columns: list, 
                 normalisation_strategy: str, 
                 model:ClassifierMixin):
        self.normalisation_columns = normalisation_columns
        self.normalisation_strategy = normalisation_strategy
        self.model = model


    def fit(self, X, y):
        """
        Fit the model training pipeline to the data.
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target variable.
        """
        logging.info("Fitting model training pipeline...")
        
        data_processing_pipeline = build_data_processing_pipeline(
            normalisation_columns=self.normalisation_columns,
            normalisation_strategy=self.normalisation_strategy
        )
        
        X_processed, y_processed = data_processing_pipeline.fit_transform(X, y)
        
        self.model.fit(X_processed, y_processed)
        logging.info("Model training completed successfully.")
        return self.model
    


def build_model_training_pipeline(normalisation_columns: list, 
                                  normalisation_strategy: str, 
                                  X:pd.DataFrame, 
                                  y:pd.Series) -> ClassifierMixin:
    """
    Build a model training pipeline with the specified normalisation columns and strategy.
    Args:
        normalisation_columns (list): List of columns to normalise.
        normalisation_strategy (str): Normalisation strategy to use ('minmax' or 'std_scaler').
    Returns:
        ModelTrainingPipeline: An instance of ModelTrainingPipeline.
    """
    logging.info("Building model training pipeline...")

    model_training_pipeline = ModelTrainingPipeline(
        normalisation_columns=normalisation_columns,
        normalisation_strategy=normalisation_strategy,
        model=RandomForestClassifier(random_state=42)
    )

    trained_model = model_training_pipeline.fit(X, y)
    return trained_model
   


if __name__ == "__main__":
    
    data_loader = DataLoader(path=config.INPUT_DATA_PATH,
                             target_column=config.TARTGET_COLUMNS[0])
    X, y = data_loader.load()

    # trained_model = build_model_training_pipeline(
    #     normalisation_columns=config.NORMALISATION_COLUMNS,
    #     normalisation_strategy=config.NORMALISATION_STRATEGY, 
    #     X=X, y=y)
    
    processing_pipeline = build_data_processing_pipeline(
        normalisation_columns=config.NORMALISATION_COLUMNS,
        normalisation_strategy=config.NORMALISATION_STRATEGY
    )
    X_processed, y_processed = processing_pipeline.fit_transform(X, y)
    model_uri = 'runs:/9cc8c83f542d49f6b1b0f01ff28d1950/model'

    input_data = X_processed

    os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/vijaytakbhate2002/credit_fraud_detection_project_with_mlflow_dagshub_dvc.mlflow"  # Set your DagsHub tracking URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "vijaytakbhate20@gmail.com" # Set your DagsHub username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = ""

    result = mlflow.models.predict(
        model_uri=model_uri,
        input_data=input_data,
        env_manager="local",
    )

    print("Prediction result:", result)
    