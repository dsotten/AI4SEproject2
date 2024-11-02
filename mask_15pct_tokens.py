import pandas as pd
import random as random
import ast

def main_csv(csv_path):
    df = pd.read_csv(csv_path)

    tokenized_functions = []
    masked_functions = []

    for index, row in df.iterrows():
        tokenized_function = ast.literal_eval(row['tokenized_function'])
        tokenized_functions.append(tokenized_function)

        masked = []
        for token in tokenized_function:
            if random.randint(1, 100) <= 15:
                masked.append('<masked>')
            else:
                masked.append(token)
        masked_functions.append(masked)

    output_df = pd.DataFrame({
        'tokenized_function': tokenized_functions,
        'masked_function': masked_functions
    })

    output_df.to_csv('15pct_masked.csv', index=False)

def main_df(df):
    # df = pd.read_csv(csv_path)

    tokenized_functions = []
    masked_functions = []

    for index, row in df.iterrows():
        tokenized = row['tokenized_function']
        for function in tokenized:
            tokenized_functions.append(function)

            masked = []
            for token in tokenized:
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
    # output_df.to_csv('15pct_masked', index=False)