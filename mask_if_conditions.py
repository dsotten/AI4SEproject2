import pandas as pd
import ast

def main_csv(csv_path):
    df = pd.read_csv(csv_path)

    tokenized_functions = []
    masked_functions = []

    for index, row in df.iterrows():
        tokenized_function = ast.literal_eval(row['tokenized_function'])
        tokenized_functions.append(tokenized_function)

        masked = []
        if_found = False
        for token in tokenized_function:
            if token == 'if':
                masked.append(token)
                masked.append('<condition>')
                if_found = True
            elif if_found:
                if token == ':':
                    masked.append(token)
                    if_found = False
            else:
                masked.append(token)

        masked_functions.append(masked)

    output_df = pd.DataFrame({
        'tokenized_function': tokenized_functions,
        'masked_function': masked_functions
    })

    output_df.to_csv('conditions_masked.csv', index=False)

def main_df(df):
    # df = pd.read_csv(csv_path)

    tokenized_functions = []
    masked_functions = []

    for index, row in df.iterrows():
        tokenized = row['tokenized_function']
        for function in tokenized:
            tokenized_functions.append(function)

            masked = []
            if_found = False
            for token in tokenized:
                if token == 'if':
                    masked.append(token)
                    masked.append('<condition>')
                    if_found = True
                elif if_found:
                    if token == ':':
                        masked.append(token)
                        if_found = False
                else:
                    masked.append(token)

            masked_functions.append(masked)

    output_df = pd.DataFrame({
        'tokenized_function': tokenized_functions,
        'masked_function': masked_functions
    })

    return output_df
    # output_df.to_csv('conditions_masked', index=False)