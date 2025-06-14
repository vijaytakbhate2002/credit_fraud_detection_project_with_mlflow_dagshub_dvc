
INPUT_DATA_PATH="data\\default_of_credit_card_clients.csv"
PROCESSED_DATA_PATH = "data\\processed_data\\processed_data.csv"

TEXT_COLUMNS = []
NUM_COLUMNS = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'default payment next month']

NORMALISATION_COLUMNS = ['LIMIT_BAL']
TARTGET_COLUMNS = ['default_payment_next_month']
NORMALISATION_STRATEGY = 'minmax'

SAMPLING_STRATEGY = 'auto'  
