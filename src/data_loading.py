import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
logging.basicConfig(
    filename='app.log',
    filemode='w',  
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



class DataLoader:
    def __init__(self, path:str, target_column:str) -> None:
        self.path = path
        self.target_column = target_column


    def __check(self) -> None:
        """
        Check if the file exists and load it into a DataFrame.
        Returns: 
            pd.DataFrame: DataFrame containing the loaded data."""
        
        if '.csv' in self.path:
            return 'csv'
        elif '.xlsx' in self.path:
            return 'xlsx'
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
        


    def load(self) -> tuple:
        """
        Load data from the specified path and split it into training and testing sets.
        Returns:
            tuple: A tuple containing the training features (X) and target variable (y).
        """
        file_type = self.__check()
        if file_type == 'csv':
            df = pd.read_csv(self.path)
        elif file_type == 'xlsx':
            df = pd.read_excel(self.path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
        
        if df.empty:
            logging.warning("The DataFrame is empty after loading.")
            X, y = pd.DataFrame(), pd.Series()
            logging.info("Returning empty DataFrame and Series.")
        else:
            X, y = df.drop(columns=[self.target_column]), df[self.target_column]
            logging.info("Data loaded and split into training and testing sets.")
        logging.info(f"Data loaded successfully from {self.path}.")
        return X, y
    


class DataSaver(DataLoader):

    def _init_(self, path: str, target_column:str) -> None:
        super().__init__(path, target_column)


    def save_df(self, df: pd.DataFrame) -> None:
        file_type = self._DataLoader__check()
        if file_type == 'csv':
            df.to_csv(self.path, index=False)
        elif file_type == 'xlsx':
            df.to_excel(self.path, index=False)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
        logging.info(f"Data saved successfully to {self.path}.")
        return None
    
    
    def save_split(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Save the training features (X) and target variable (y) to the specified path.
        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target variable.
        """
        file_type = self._DataLoader__check()
        if file_type == 'csv':
            X.to_csv(self.path, index=False)
            y.to_csv(self.path.replace('.csv', '_target.csv'), index=False)
        elif file_type == 'xlsx':
            df = pd.concat([X, y], axis=1)
            df.to_excel(self.path, index=False)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
        logging.info(f"Data split saved successfully to {self.path}.")
        return None
    
