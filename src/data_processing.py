import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import config
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(
    filename='app.log',
    filemode='a',  
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
    



class NullHandler(BaseEstimator, TransformerMixin):

    def fit(self, X:pd.DataFrame, y:pd.Series) -> None:
        self.df = pd.concat([X, y], axis='columns') if y is not None else X
        if X.isnull().values.any():
            logging.warning("Null values found in the DataFrame.")
        else:
            logging.info("No null values found in the DataFrame.")
        return self



    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Transform method fills null values with 0 and return X, y as tuple"""
        logging.info("Removing null valued rows from X...")
        df = self.df.dropna()
        if y is not None:
            self.X = df.drop(columns=y.name)
            self.y = df[y.name]
        else:
            logging.warning("No target variable provided, returning only features...")
            self.X = df
            self.y = None
        return self.X, self.y




class DataProcessor(BaseEstimator, TransformerMixin):


    def __init__(self, normalisation_columns:list, normalisation_strategy:str='minmax'):
        self.normalisation_strategy = normalisation_strategy

        if normalisation_strategy != 'minmax':
            raise ValueError("Currently only 'minmax' normalisation strategy is supported.")
        self.normalisation_columns = normalisation_columns



    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        logging.info("Fitting DataProcessor...")

        if self.normalisation_strategy == 'minmax':
            self.scaler = MinMaxScaler()

        elif self.normalisation_strategy == 'standard':
            self.scaler = StandardScaler()
            raise NotImplementedError("Standard normalisation is not implemented yet.")
        
        return self



    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Transform method scales the DataFrame using MinMaxScaler and returns X, y as tuple.
        If y is provided, it will be returned as well."""
        self.X = X.copy()
        self.y = y.copy() if y is not None else None
        for col in self.normalisation_columns:
            self.X[col] = self.scaler.fit_transform(self.X[[col]])
        logging.info(f"Normalised columns: {self.normalisation_columns} using {self.normalisation_strategy} strategy.")
        return self.X, self.y




class DataSampler(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state



    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        logging.info("Fitting DataSampling...")
        self.smote = SMOTE(random_state=self.random_state)
        return self



    def transform(self, X:pd.DataFrame, y:pd.Series) -> tuple:
        """
        Oversample the minority class in training data using SMOTE.
        This method returns the resampled DataFrame and Series."""
        logging.info("Transforming DataSampling...")
        
        X_train_resampled, y_train_resampled = self.smote.fit_resample(X, y)
        
        logging.info(f"Class distribution before resampling: {pd.Series(y).value_counts(normalize=True)}")
        logging.info(f"Class distribution after resampling: {pd.Series(y_train_resampled).value_counts(normalize=True)}")

        return X_train_resampled, y_train_resampled
