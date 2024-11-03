import ast
import pandas as pd
import random
from transformers import LlamaTokenizer

# Initialize tokenizer
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")

def tokenize_function(code, max_length=2048):
    """
    Tokenizes the input Python function using LLaMA tokenizer.
    Returns the list of tokenized IDs.
    """
    try:
        tokens = tokenizer(code, return_tensors="pt", truncation=True, max_length=max_length)
        return tokens['input_ids'].tolist()[0]
    except Exception as e:
        print(f"Tokenization error for code: {code}\nError: {e}")
        return None

def extract_functions_with_if_statements(code):
    """
    Parses the given Python code and extracts function definitions
    that contain at least one 'if' statement.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # Return an empty list if the code has syntax errors
    
    functions_with_if = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if any(isinstance(child, ast.If) for child in ast.walk(node)):
                try:
                    function_source = ast.unparse(node)
                    functions_with_if.append(function_source)
                except Exception as e:
                    print(f"Unparsing error for function node: {node}\nError: {e}")
                    continue
    return functions_with_if

def mask_if_statements(code):
    """
    Masks 'if' statements in the function code by replacing the condition with '<MASK>'.
    Returns the masked code and the original condition for labels.
    """
    try:
        tree = ast.parse(code)
        original_conditions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                original_conditions.append(ast.unparse(node.test))  # Capture original condition
                node.test = ast.Constant(value="<MASK>", kind=None)  # Mask the condition
        masked_code = ast.unparse(tree)
        
        return masked_code, original_conditions  # Return masked code and original conditions
    except Exception as e:
        print(f"Masking error for code: {code}\nError: {e}")
        return code, []

def random_token_masking(code, mask_token="<MASK>", mask_ratio=0.15):
    """
    Masks a percentage of tokens in the code with a specified mask token.
    Returns the masked code and the original tokens for labels.
    """
    # Tokenize the code using the tokenizer
    tokenized = tokenizer.tokenize(code)
    total_tokens = len(tokenized)
    
    # Determine how many tokens to mask
    num_to_mask = max(1, int(total_tokens * mask_ratio))  # Ensure at least one token is masked

    # Randomly select indices to mask
    indices_to_mask = random.sample(range(total_tokens), num_to_mask)

    # Create masked tokens
    masked_tokens = [tokenized[i] if i not in indices_to_mask else mask_token for i in range(total_tokens)]

    return masked_tokens, [tokenized[i] for i in range(total_tokens) if i not in indices_to_mask]
    
def extract_and_tokenize_functions_from_csv(csv_file, column_name, pretrain_output, finetune_output):
    df = pd.read_csv(csv_file)
    pretrain_data = []
    finetune_data = []

    for _, row in df.iterrows():
        code = row[column_name]
        functions_with_if = extract_functions_with_if_statements(code)

        for func in functions_with_if:
            # Pre-training: Random token masking
            pretrain_masked_func, original_tokens = random_token_masking(func)  # Use the masking function
            if pretrain_masked_func is not None:
                labels = tokenizer.convert_tokens_to_ids(original_tokens)  # Use original tokens for labels
                pretrain_data.append({
                    'input_ids': pretrain_masked_func,
                    'labels': labels
                })

            # Fine-tuning: Masking only 'if' conditions
            finetune_masked_func, original_conditions = mask_if_statements(func)
            tokenized_finetune = tokenize_function(finetune_masked_func)
            if tokenized_finetune is not None:
                labels_finetune = labels_finetune = [tokenizer.convert_tokens_to_ids(condition) for condition in original_conditions]
                finetune_data.append({
                    'input_ids': tokenized_finetune,
                    'labels': labels_finetune
                })

    pretrain_df = pd.DataFrame(pretrain_data)
    pretrain_df.to_csv(pretrain_output, index=False)

    finetune_df = pd.DataFrame(finetune_data)
    finetune_df.to_csv(finetune_output, index=False)

# Example usage
csv_file = 'full.csv'
column_name = 'code'
pretrain_output = 'pretrain_dataset.csv'
finetune_output = 'finetune_dataset.csv' 

extract_and_tokenize_functions_from_csv(csv_file, column_name, pretrain_output, finetune_output)