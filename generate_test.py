from transformers import T5ForConditionalGeneration, LlamaTokenizer
import pandas as pd
import torch

# Generate Test
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
model = T5ForConditionalGeneration.from_pretrained('./finetuned_model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_data = pd.read_csv(finetune_test_dataset.csv)

def generate_test_results(test_data, model):
    """
    Generates test results for the provided test dataset.
    
    Args:
        test_data (list of dict): The list containing input data for testing.
        model: The trained model for prediction.
    
    Returns:
        pd.DataFrame: A DataFrame containing test results.
    """
    results = []

    for item in test_data:
        input_code = item['input_method']
        expected_condition = item['target_block']
        
        # Predict the if condition using the model
        predicted_condition, prediction_score = model.predict(input_code)
        
        # Determine if the prediction is correct
        correct_prediction = predicted_condition == expected_condition[0]
        
        # Append result to the list
        results.append({
            'Input provided to the model': input_code,
            'Correct prediction (true/false)': correct_prediction,
            'Expected if condition': expected_condition[0] if expected_condition else None,
            'Predicted if condition': predicted_condition,
            'Prediction score (0-100)': prediction_score
        })
    
    return pd.DataFrame(results)

# Load test datasets
generated_test_data = pd.read_csv('finetune_test_dataset.csv')
provided_test_data = pd.read_csv('provided_test_data.csv')


# Generate results for generated test dataset
generated_results_df = generate_test_results(generated_test_data.to_dict(orient='records'), model)
generated_results_df.to_csv('generated-testset.csv', index=False)

# Generate results for provided test dataset
provided_results_df = generate_test_results(provided_test_data.to_dict(orient='records'), model)
provided_results_df.to_csv('provided-testset.csv', index=False)


# input_texts = test_data['input_column'].tolist()
# predictions = []

# for text in input_texts:
#     input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)  # Move to device if using GPU
#     outputs = model.generate(input_ids)
#     prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     predictions.append(prediction)

# # Add predictions to your DataFrame
# test_data['predictions'] = predictions

# # Save the predictions to a new CSV file
# test_data.to_csv('test_results.csv', index=False)
# print('Test results saved to test_results.csv.')