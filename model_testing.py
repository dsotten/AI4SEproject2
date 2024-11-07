import ast
import pandas as pd
import extract_and_mask_if2
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TFAutoModel, LlamaTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy
# import tensorflow as tf

# model_path = "./finetuned_model/model.safetensors" #path/to/your/model/or/name/on/hub
model_path = "./finetuned_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = T5ForConditionalGeneration.from_pretrained(model_path, device_map=device)
# tokenizer = T5Tokenizer.from_pretrained(model_path)
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")

# inputs = tokenizer.encode("This movie was really", return_tensors="pt")
# outputs = model.generate(inputs)
# print(tokenizer.decode(outputs[0]))

generated_test_csv = 'finetune_test_dataset.csv'
provided_test_csv = 'sample.csv'
generated_test_df = pd.read_csv(generated_test_csv)
provided_test_df = pd.read_csv(provided_test_csv)

generated_test_df['input_ids'] = generated_test_df['input_ids'].apply(ast.literal_eval)
generated_test_df['target_block'] = generated_test_df['target_block'].apply(ast.literal_eval)

generated_testset_results = []
provided_testset_results = []

num_preds = 0
correct_preds = 0

print("Beginning sample dataset testing")

for _, row in provided_test_df.iterrows():
    # model_input = tokenizer.encode(row['input_method'])
    model_input = tokenizer(row['input_method'], return_tensors="pt", truncation=True, max_length=2048).input_ids
    expected_if = row['target_block']
    pred_correct = False
    num_preds += 1

    #How do we actually call the model?
    predicted_if = model.generate(model_input.cuda())
    # pred_str = ''
    predicted_if = tokenizer.decode(predicted_if[0], skip_special_tokens=True)
    # predicted_if = tokenizer.decode(predicted_if[0])
    # for i in predicted_if:
    #         pred_str += tokenizer.decode(predicted_if[i], skip_special_tokens=True)

    if (expected_if == predicted_if):
        pred_correct = True
        correct_preds += 1

    provided_testset_results.append({
        'model_input': row['input_method'],
        'pred_correct': pred_correct,
        'expected_if': expected_if,
        'predicted_if': predicted_if,
        'pred_score': correct_preds/num_preds
    })
pd.DataFrame(provided_testset_results).to_csv('provided-testset.csv', index=False)

num_preds = 0
correct_preds = 0

print("Beginning generated dataset testing")

for _, row in generated_test_df.iterrows():
    # model_input = tokenizer(row['input_ids'], return_tensors="pt", truncation=True, max_length=2048)
    # model_input = row['input_ids']
    model_input = torch.tensor(row['input_ids']).unsqueeze(0)
    expected_if = row['target_block']
    pred_correct = False
    num_preds += 1

    # generated_test_df2[_][row['input_ids']]

    #How do we actually call the model?
    predicted_if = model.generate(model_input.cuda())
    predicted_if = tokenizer.decode(predicted_if[0], skip_special_tokens=True)

    if (expected_if == predicted_if):
        pred_correct = True
        correct_preds += 1

    generated_testset_results.append({
        'model_input': generated_test_df.loc[_, 'input_method'],
        'pred_correct': pred_correct,
        'expected_if': expected_if,
        'predicted_if': predicted_if,
        'pred_score': correct_preds/num_preds
    })
pd.DataFrame(generated_testset_results).to_csv('generated-testset.csv', index=False)



