import ast
import pandas as pd
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")


def tokenize_function(code):
    """
    Tokenizes the input Python function using LLaMA tokenizer.
    Returns the list of tokenized IDs.
    """
    tokens = tokenizer(code, return_tensors="pt")
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
    
def mask_if_statements(code):
    """Masks 'if' statements in the function code by replacing the condition with '<MASK>'."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            node.test = ast.Constant(value="<MASK>", kind=None)
    try:
        return ast.unparse(tree)
    except Exception:
        return code


# Extract if functions, and tokenize them
def extract_and_tokenize_functions_from_csv(csv_file, column_name, output_file):
    """Extracts functions with 'if' statements, tokenizes, masks them, and saves in a new CSV file."""
    df = pd.read_csv(csv_file)
    
    original_functions = []
    tokenized_functions = []
    masked_functions = []
    tokenized_masked_functions = []
    
    for index, row in df.iterrows():
        code = row[column_name]
        functions_with_if = extract_functions_with_if_statements(code)
        
        for func in functions_with_if:
            original_functions.append(func)
            
            try:
                # Tokenize the original function
                tokenized_code = tokenize_function(func)
                tokenized_functions.append(tokenized_code)
                
                # Mask if-statements and add to masked functions list
                masked_func = mask_if_statements(func)
                masked_functions.append(masked_func)
                
                # Tokenize the masked function
                tokenized_masked_code = tokenize_function(masked_func)
                tokenized_masked_functions.append(tokenized_masked_code)
                
            except Exception as e:
                print(f"Error processing function: {func}\nError: {e}")
                continue
    
    output_df = pd.DataFrame({
        'original_function': original_functions,
        'tokenized_function': tokenized_functions,
        'masked_function': masked_functions,
        'tokenized_masked_function': tokenized_masked_functions
    })
    
    output_df.to_csv(output_file, index=False)

# Example usage
csv_file = 'full.csv'  # Path to your CSV file
column_name = 'code'  # The name of the column containing Python code
output_file = 'tokenized_and_masked_functions.csv'  # The name of the output CSV file

extract_and_tokenize_functions_from_csv(csv_file, column_name, output_file)

