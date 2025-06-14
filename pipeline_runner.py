import pandas as pd
import logging  
import config
from src.data_loading import DataLoader, DataSaver
from data_processing_pipeline import build_data_processing_pipeline


if __name__ == "__main__":
    print("Building data processing pipeline...")
    X, y = DataLoader(path=config.INPUT_DATA_PATH,  target_column = config.TARTGET_COLUMNS[0]).load()

    logging.info("Building data processing pipeline...")
    
    pipeline = build_data_processing_pipeline(config.NORMALISATION_COLUMNS, 
                                              config.NORMALISATION_STRATEGY)
    
    X_processed, y_processed = pipeline.fit_transform(X, y)

    data_dumper = DataSaver(path=config.PROCESSED_DATA_PATH, 
                            target_column=config.TARTGET_COLUMNS[0])
    
    data_dumper.save_df(pd.concat([X_processed, y_processed], axis=1))
    logging.info("Data processing is done and saved to the specified path...")
    