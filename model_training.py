import loaddata
import extract_if
import mask_15pct_tokens
import mask_if_conditions
import pandas as pd

import os.path
from pathlib import Path

from transformers import T5ForConditionalGeneration
from transformers import LlamaTokenizer
from transformers import Trainer
from datasets import Dataset 
# import torch.cuda
# import torch

print('Start')

model_base = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
skipDatasetCreation = True

#Tokenize full dataset
dataset_filepath = 'python/final/jsonl/test/python_test_1.jsonl'
csv_file = 'full.csv'  # Path to your CSV file
column_name = 'code'  # The name of the column containing Python code
tokenized_csv = 'tokenized_functions.csv'  # The name of the output CSV file
# tokenized_df = extract_if.extract_and_tokenize_functions_from_df(pd.read_csv(csv_file), column_name)

if Path(csv_file).is_file():
    pass
else:
    loaddata.main(dataset_filepath)

if Path(tokenized_csv).is_file():
    print('Tokenized dataset found, skipping tokenization process')
    pass
else:
    extract_if.extract_and_tokenize_functions_from_csv(csv_file, column_name, tokenized_csv)
# tokenized_df = pd.read_csv(tokenized_csv)

print('Full dataset tokenized')

#Splits a CSV file into two datasets based on percentages.
def split_csv_by_percentage(csv_file, output_path1, output_path2, pct1=0.7, pct2=0.3):
    
    df = pd.read_csv(csv_file)
    total_rows = len(df)

    size1 = int(total_rows * pct1)
    size2 = int(total_rows * pct2)

    df1 = df.iloc[:size1]
    df2 = df.iloc[size1:size1 + size2]

    # return [df1, df2]
    df1.to_csv(output_path1, index=False)
    df2.to_csv(output_path2, index=False)

pretrain_csv = 'pretrain.csv'
ft_tosplit_csv = 'ft_tosplit.csv'
finetune_csv = 'finetune.csv'
evaluate_csv = 'evaluate.csv'

#Set up pretraining dataset
if Path(pretrain_csv).is_file() and Path(ft_tosplit_csv).is_file():
    print('Pretrain dataset found, skipping dataset split')
    pass
else:
    # dataframes = split_csv_by_percentage(tokenized_df, csv_file1, csv_file2)
    split_csv_by_percentage(tokenized_csv,pretrain_csv,ft_tosplit_csv)
# pretrain_set = Dataset.from_pandas(mask_15pct_tokens.main_df(pd.read_csv(pretrain_csv)))

if Path('15pct_masked.csv').is_file():
    print('15 percent of tokens have already been masked')
    pass
else:
    mask_15pct_tokens.main_csv(pretrain_csv)
pretrain_set = Dataset.from_pandas(pd.read_csv('15pct_masked.csv'))

print('Pretraining dataset created')

if Path('conditions_masked.csv').is_file():
    # dataframes = split_csv_by_percentage(finetune_csv, finetune_csv, evaluate_csv, 0.85, 0.15)
    print('If conditions have already been masked')
    pass
else:
    mask_if_conditions.main_csv(ft_tosplit_csv)

if Path(finetune_csv).is_file() and Path(evaluate_csv).is_file():
    pass
else:
    split_csv_by_percentage('conditions_masked.csv', finetune_csv, evaluate_csv, 0.85, 0.15)
finetune_set = Dataset.from_pandas(pd.read_csv(finetune_csv))
eval_set = Dataset.from_pandas(pd.read_csv(evaluate_csv))

print('Finetuning and Evaluation datasets created')

model_trainer1 = Trainer(model_base, train_dataset=pretrain_set, eval_dataset=eval_set)
model1 = model_trainer1.train()

print('Pretraining step complete')

model_trainer2 = Trainer(model1, train_dataset=finetune_set, eval_dataset=eval_set)
eval1 = model_trainer2.evaluate()
model2 = model_trainer2.train()

print('Finetuning step complete')

model_final_eval = Trainer(model2, train_dataset=finetune_set, eval_dataset=eval_set)
eval2 = model_final_eval.evaluate()

# text = #kate's_dataset_creator.random_sample()
# input_ids = tokenizer(text, return_tensors="pt").input_ids

# # simply generate a single sequence
# generated_ids = model2.generate(input_ids, max_length=10)
# print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))