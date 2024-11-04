import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./finetuned_model" #path/to/your/model/or/name/on/hub
# device = "cpu" # or "cuda" if you have a GPU

model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path)

# inputs = tokenizer.encode("This movie was really", return_tensors="pt")
# outputs = model.generate(inputs)
# print(tokenizer.decode(outputs[0]))

generated_test_csv = 'finetune_test_dataset.csv'
provided_test_csv = 'sample.csv'
generated_test_df = pd.DataFrame(generated_test_csv)
provided_test_df = pd.DataFrame(provided_test_csv)

generated_testset_results = []
provided_testset_results = []
# pred_score = 0
# pred_calc = 

for _, row in generated_test_df.iterrows():
    model_input = tokenizer.encode(row['input_method'])
    expected_if = row['target_block']
    pred_correct = False

    #How do we actually call the model?
    predicted_if = model.generate(model_input)

    if (expected_if == predicted_if):
        pred_correct = True

    generated_testset_results.append({
        'model_input': model_input,
        'pred_correct': pred_correct,
        'expected_if': expected_if,
        'predicted_if': predicted_if,
        'pred_score':
    })
pd.DataFrame(generated_testset_results).to_csv('generated-testset', index=False)

for _, row in provided_test_df.iterrows():
    model_input = tokenizer.encode(row['input_method'])
    expected_if = row['target_block']
    pred_correct = False

    #How do we actually call the model?
    predicted_if = model.generate(model_input)

    if (expected_if == predicted_if):
        pred_correct = True

    provided_testset_results.append({
        'model_input': model_input,
        'pred_correct': pred_correct,
        'expected_if': expected_if,
        'predicted_if': predicted_if,
        'pred_score': 
    })
pd.DataFrame(provided_testset_results).to_csv('provided-testset', index=False)