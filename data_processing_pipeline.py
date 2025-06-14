import pandas as pd
import numpy as np
import logging  
from sklearn.pipeline import Pipeline
from src.data_processing import NullHandler, DataProcessor, DataSampler
from src.data_loading import DataLoader, DataSaver
import config

class DataProcessingPipeline:
    def __init__(self, steps:list):
        self.steps = steps

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DataProcessingPipeline':
        """
        Fit the pipeline to the data.
        """
        for name, step in self.steps:
            logging.info(f"Fitting step: {name}")
            if hasattr(step, 'fit'):
                step.fit(X, y)
            else:
                raise ValueError(f"Step {name} does not have a fit method.")
        return self
    
    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> tuple:
        """
        Transform the data using the fitted pipeline.
        """
        for name, step in self.steps:
            logging.info(f"Transforming step: {name}")
            if hasattr(step, 'transform'):
                X, y = step.transform(X, y)
            else:
                raise ValueError(f"Step {name} does not have a transform method.")
        return X, y

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> tuple:
        """
        Fit the pipeline to the data and then transform it.
        """
        self.fit(X, y)
        return self.transform(X, y)


def build_data_processing_pipeline(normalisation_columns:list, normalisation_strategy:str) -> DataProcessingPipeline:
    """
    Build a data processing pipeline with the specified normalisation columns and strategy.
    Args:
        normalisation_columns (list): List of columns to normalise.
        normalisation_strategy (str): Normalisation strategy to use ('minmax' or 'std_scaler').
    """
    logging.info("Building data processing pipeline...")
    
    # Define the steps in the pipeline
    steps = [
       ('null_handler', NullHandler()),
        ('data_processor', DataProcessor(normalisation_columns=normalisation_columns, 
                      normalisation_strategy=normalisation_strategy)),
       ('data_sampler', DataSampler())
    ]
    
    # Create the pipeline
    pipeline = DataProcessingPipeline(steps=steps)
    
    logging.info("Data processing pipeline built successfully.")
    return pipeline








