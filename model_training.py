# import loaddata
import extract_if
import mask_15pct_tokens
import mask_if_conditions
import pandas as pd

from transformers import T5ForConditionalGeneration
from transformers import LlamaTokenizer
from transformers import Trainer
from datasets import Dataset 
import torch.cuda
import torch

model_base = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")

#Tokenize full dataset
csv_file = 'full.csv'  # Path to your CSV file
column_name = 'code'  # The name of the column containing Python code
# output_file = 'tokenized_functions.csv'  # The name of the output CSV file
tokenized_df = extract_if.extract_and_tokenize_functions_from_df(pd.read_csv(csv_file), column_name)

print('')
print('Full dataset tokenized')
print('')

#Splits a CSV file into two datasets based on percentages.
def split_csv_by_percentage(df, output_path1, output_path2, pct1=0.7, pct2=0.3):
    
    # df = pd.read_csv(csv_file)
    total_rows = len(df)

    size1 = int(total_rows * pct1)
    size2 = int(total_rows * pct2)

    df1 = df.iloc[:size1]
    df2 = df.iloc[size1:size1 + size2]

    return [df1, df2]
    # df1.to_csv(f"{output_path1}.csv", index=False)
    # df2.to_csv(f"{output_path2}.csv", index=False)

csv_file1 = 'pretrain'
csv_file2 = 'finetune'
csv_file3 = 'evaluate'

#Set up pretraining dataset
dataframes = split_csv_by_percentage(tokenized_df, csv_file1, csv_file2)
pretrain_set = Dataset.from_pandas(mask_15pct_tokens.main_df(dataframes[1],'code'))
finetune_set = mask_if_conditions.main_df(dataframes[2])

print('')
print('Pretraining dataset created')
print('')

dataframes = split_csv_by_percentage(finetune_set, csv_file2, csv_file3, 0.85, 0.15)
finetune_set = Dataset.from_pandas(dataframes[1])
eval_set = Dataset.from_pandas(dataframes[2])

print('')
print('Finetuning and Evaluation datasets created')
print('')

model_trainer1 = Trainer(model_base, train_dataset=pretrain_set, eval_dataset=eval_set)
model1 = model_trainer1.train

print('')
print('Pretraining step complete')
print('')

model_trainer2 = Trainer(model1, train_dataset=finetune_set, eval_dataset=eval_set)
eval1 = model_trainer2.evaluate
model2 = model_trainer2.train

print('')
print('Finetuning step complete')
print('')

model_final_eval = Trainer(model2, train_dataset=finetune_set, eval_dataset=eval_set)
eval2 = model_final_eval.evaluate

# text = #kate's_dataset_creator.random_sample()
# input_ids = tokenizer(text, return_tensors="pt").input_ids

# # simply generate a single sequence
# generated_ids = model2.generate(input_ids, max_length=10)
# print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))