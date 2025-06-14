# from src.data_loading import loadData, dumpData

# df = loadData("data\\default_of_credit_card_clients.csv")
# print(df)

# dumpData(df=df, path="data\\test_path.csv")


from abc import ABC, abstractmethod
import pandas as pd

# Step 1: Abstract Base Class
class PipelineStep(ABC):
    @abstractmethod
    def process(self, data):
        pass

class DataLoader(PipelineStep):
    def process(self, data=None):
        df = pd.read_csv('data\\default_of_credit_card_clients.csv')
        return df

class DataCleaner(PipelineStep):
    def process(self, df):
        df = df.dropna()
        return df

class DataTransformer(PipelineStep):
    def process(self, df):
        df['default payment next month'] = df['default payment next month'].astype('category')
        return df

class DataSaver(PipelineStep):
    def process(self, df):
        df.to_csv('data\\processed_data\\cleaned_data.csv', index=False)
        return df


class DataPipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self):
        data = None
        for step in self.steps:
            data = step.process(data)
        return data

# Instantiate steps
pipeline = DataPipeline([
    DataLoader(),
    DataCleaner(),
    DataTransformer(),
    DataSaver()
])

pipeline.run()
