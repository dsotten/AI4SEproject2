import pandas as pd
import random as random

def main_csv(csv_path, column_name):
    df = pd.read_csv(csv_path)

    tokenized_functions = []
    masked_functions = []

    for index, row in df.iterrows():
        tokenized = row['tokenized_function']
        for function in tokenized:
            tokenized_functions.append(function)
            tokens = tokenized.split(' ')

            masked = []
            for token in tokens:
                if random.randint(1, 100) <= 15:
                    masked.append('<masked>')
                else:
                    masked.append(token)
            masked_functions.append(masked)

    output_df = pd.DataFrame({
        'tokenized_function': tokenized_functions,
        'masked_function': masked_functions
    })

    output_df.to_csv('15pct_masked', index=False)

def main_df(df, column_name):
    tokenized_functions = []
    masked_functions = []

    for index, row in df.iterrows():
        tokenized = row['tokenized_function']
        for function in tokenized:
            tokenized_functions.append(function)
            tokens = tokenized.split(' ')

            masked = []
            for token in tokens:
                if random.randint(1, 100) <= 15:
                    masked.append('<masked>')
                else:
                    masked.append(token)
            masked_functions.append(masked)

    output_df = pd.DataFrame({
        'tokenized_function': tokenized_functions,
        'masked_function': masked_functions
    })

    return output_df