import ast
import pandas as pd
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")


def tokenize_function(code):
    """
    Tokenizes the input Python function using LLaMA tokenizer.
    Returns the list of tokenized IDs.
    """
    tokens = tokenizer(code, return_tensors="pt", truncation=True)
    return tokens['input_ids'].tolist()[0]
    

def extract_functions_with_if_statements(code):
    """
    Parses the given Python code and extracts function definitions
    that contain at least one 'if' statement.
    """
    # Parse the code into an AST
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # Return an empty list if the code has syntax errors
    
    # List to store functions with 'if' statements
    functions_with_if = []
    
    # Traverse the tree
    for node in ast.walk(tree):
        # Check if the node is a function definition
        if isinstance(node, ast.FunctionDef):
            # Check if the function contains an 'if' statement
            for child in ast.walk(node):
                if isinstance(child, ast.If):
                    # Extract the function source code
                    try:
                        function_source = ast.unparse(node)
                    except Exception as e:
                        continue
                    functions_with_if.append(function_source)
                    break
    
    return functions_with_if


# Extract if functions, and tokenize them
def extract_and_tokenize_functions_from_csv(csv_file, column_name, output_file):
    """
    Reads a CSV file, extracts functions from the specified column
    that contain 'if' statements, tokenizes them using LLaMA, and saves
    the result in a new CSV file.
    """
    df = pd.read_csv(csv_file)
    
    # Lists to store the original and tokenized functions
    original_functions = []
    tokenized_functions = []
    
    for index, row in df.iterrows():
        code = row[column_name]
        functions_with_if = extract_functions_with_if_statements(code)
        
        for func in functions_with_if:
            original_functions.append(func)
            try:
                tokenized_code = tokenize_function(func)
                tokenized_functions.append(tokenized_code)
            except Exception as e:
                print(f"Error tokenizing function: {func}\nError: {e}")
                continue
    
    # Create a DataFrame to store the results
    output_df = pd.DataFrame({
        'original_function': original_functions,
        'tokenized_function': tokenized_functions
    })
    
    # Save the DataFrame to a CSV file
    output_df.to_csv(output_file, index=False)

def extract_and_tokenize_functions_from_df(df, column_name):
    """
    Reads a dataframe, extracts functions from the specified column
    that contain 'if' statements, tokenizes them using LLaMA, and returns
    the resulting dataframe.
    """
    # Lists to store the original and tokenized functions
    original_functions = []
    tokenized_functions = []
    
    for index, row in df.iterrows():
        code = row[column_name]
        functions_with_if = extract_functions_with_if_statements(code)
        
        for func in functions_with_if:
            original_functions.append(func)
            try:
                tokenized_code = tokenize_function(func)
                tokenized_functions.append(tokenized_code)
            except Exception as e:
                print(f"Error tokenizing function: {func}\nError: {e}")
                continue
    
    # Create a DataFrame to store the results
    output_df = pd.DataFrame({
        'original_function': original_functions,
        'tokenized_function': tokenized_functions
    })
    
    # Return the DataFrame
    return output_df

# Example usage
# csv_file = 'full.csv'  # Path to your CSV file
# column_name = 'code'  # The name of the column containing Python code
# output_file = 'tokenized_functions.csv'  # The name of the output CSV file

# extract_and_tokenize_functions_from_csv(csv_file, column_name, output_file)

