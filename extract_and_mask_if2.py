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
    Masks a percentage of tokens in the tokenized code with a specified mask token.
    Returns the masked tokens and the corresponding labels.
    """
    # Tokenize the code using the tokenizer
    tokenized = tokenizer.tokenize(code)
    total_tokens = len(tokenized)

    # Determine how many tokens to mask
    num_to_mask = max(1, int(total_tokens * mask_ratio))  # Ensure at least one token is masked

    # Randomly select indices to mask
    indices_to_mask = random.sample(range(total_tokens), num_to_mask)

    # Create masked tokens and corresponding labels
    masked_tokens = []
    labels = []

    for i in range(total_tokens):
        if i in indices_to_mask:
            masked_tokens.append(mask_token)  # Add the mask token
            labels.append(tokenizer.convert_tokens_to_ids(tokenized[i]))  # Store the original token ID
        else:
            masked_tokens.append(tokenized[i])  # Keep the original token
            labels.append(-100)  # Use -100 for unmasked tokens

    return masked_tokens, labels
    
def extract_and_tokenize_functions_from_csv(csv_file, column_name, pretrain_output, finetune_train_output, finetune_eval_output, finetune_test_output):
    df = pd.read_csv(csv_file)
    pretrain = df[:150000]
    finetune = df[150000:200000]
    pretrain_data = []
    finetune_data = []

    for _, row in pretrain.iterrows():
        code = row[column_name]
        functions_with_if = extract_functions_with_if_statements(code)

        for func in functions_with_if:
            # Pre-training: Random token masking
            pretrain_masked_func, original_labels = random_token_masking(func)  # Use the masking function
            if pretrain_masked_func is not None:
                input_ids = tokenizer.convert_tokens_to_ids(pretrain_masked_func)
                
                pretrain_data.append({
                    'input_ids': input_ids,
                    'labels': original_labels
                })
    
    for _, row in finetune.iterrows():
        code = row[column_name]
        functions_with_if = extract_functions_with_if_statements(code)

        for func in functions_with_if:
            # Fine-tuning: Masking only 'if' conditions
            finetune_masked_func, original_conditions = mask_if_statements(func)
            tokenized_finetune = tokenize_function(finetune_masked_func)
            if tokenized_finetune is not None:
                labels_finetune = [-100] * len(tokenized_finetune)  # Default to -100 for masked tokens
                for condition in original_conditions:
                    condition_tokens = tokenizer.tokenize(condition)
                    condition_ids = tokenizer.convert_tokens_to_ids(condition_tokens)
        
                    # Ensure replace the position of <MASK> in labels_finetune
                    for i in range(len(tokenized_finetune)):
                        if tokenized_finetune[i] == tokenizer.convert_tokens_to_ids("<MASK>"):  # Check for the mask token
                            labels_finetune[i] = condition_ids[0]  # Assign the first condition's token ID
                finetune_data.append({
                    'input_ids': tokenized_finetune,
                    'labels': labels_finetune,
                    'original_method': func,
                    'input_method': finetune_masked_func,
                    'target_block': original_conditions
                })

    pretrain_df = pd.DataFrame(pretrain_data)
    pretrain_df.to_csv(pretrain_output, index=False)
    
    random.shuffle(finetune_data)
    total_finetune = len(finetune_data)
    train_size = int(total_finetune * 0.8)
    eval_size = int(total_finetune * 0.1)

    finetune_train_data = finetune_data[:train_size]
    finetune_eval_data = finetune_data[train_size:train_size + eval_size]
    finetune_test_data = finetune_data[train_size + eval_size:]

    finetune_train_df = pd.DataFrame(finetune_train_data)[['input_ids', 'labels']]
    finetune_train_df.to_csv(finetune_train_output, index=False)

    finetune_eval_df = pd.DataFrame(finetune_eval_data)[['input_ids', 'labels']]
    finetune_eval_df.to_csv(finetune_eval_output, index=False)

    # Create finetune test DataFrame with all columns and save
    finetune_test_df = pd.DataFrame(finetune_test_data)
    finetune_test_df.to_csv(finetune_test_output, index=False)

# Example usage
# csv_file = 'full.csv'
# column_name = 'code'
# pretrain_output = 'pretrain_dataset.csv'
# finetune_train_output = 'finetune_train_dataset.csv'
# finetune_eval_output = 'finetune_eval_dataset.csv'
# finetune_test_output = 'finetune_test_dataset.csv'

# extract_and_tokenize_functions_from_csv(csv_file, column_name, pretrain_output, finetune_train_output, finetune_eval_output, finetune_test_output)
