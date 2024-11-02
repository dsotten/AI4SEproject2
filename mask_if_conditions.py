import pandas as pd

def main(csv_path, column_name):
    df = pd.read_csv(csv_path)

    tokenized_functions = []
    masked_functions = []

    for index, row in df.iterrows():
        tokenized = row['tokenized_function']
        for function in tokenized:
            tokenized_functions.append(function)
            tokens = tokenized.split(' ')

            masked = []
            if_found = False
            for token in tokens:
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

    output_df.to_csv('conditions_masked', index=False)