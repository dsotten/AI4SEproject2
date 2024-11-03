import loaddata
import extract_and_mask_if2
import mask_15pct_tokens
import mask_if_conditions
import pandas as pd
import ast
import torch

from pathlib import Path

from transformers import TrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import DataCollatorWithPadding

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import LlamaTokenizer
from transformers import Trainer
from datasets import Dataset
import torch.cuda
import torch
from torch.utils.data import DataLoader

#Splits a CSV file into two datasets based on percentages.
def split_csv_by_percentage(csv_file, output_path1, output_path2=None, pct1=0.7, pct2=0.3, start_pct=0.0):
    
    df = pd.read_csv(csv_file)
    total_rows = len(df)

    size1 = int(total_rows * pct1)
    size2 = int(total_rows * pct2)
    start = int(total_rows * start_pct)

    df1 = df.iloc[start:size1 + start]
    df2 = df.iloc[size1 + start:size1 + size2 + start]

    # Output CSV files
    df1.to_csv(output_path1, index=False)
    if output_path2:
        df2.to_csv(output_path2, index=False)

# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self._data = data

#     def __len__(self):
#         return len(self._data)

#     def __getitem__(self, idx):
#         # Check if idx is a list (for batching)
#         if isinstance(idx, list):
#             # If a batch of indices is provided, return the items for those indices
#             return [self._data[i] for i in idx]
#         else:
#             # Return a single item
#             item = self._data[idx]
#             return {
#                 'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
#                 'labels': torch.tensor(item['labels'], dtype=torch.long)
#             }

# print('Start')

# tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
model_base = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_base, padding="longest")

#Tokenize full dataset
dataset_filepath = 'python/final/jsonl/test/python_test_1.jsonl'
csv_file = 'full.csv'
column_name = 'code'
pretrain_csv = 'pretrain.csv'
ft_tosplit_csv = 'ft_tosplit.csv'
finetune_csv = 'finetune.csv'
evaluate_csv = 'evaluate.csv'


if Path(csv_file).is_file():
    pass
else:
    loaddata.main(dataset_filepath)

if Path(pretrain_csv).is_file() and Path(finetune_csv).is_file():
    print('Datasets found, skipping tokenization and masking')
    pass
else:
    extract_and_mask_if2.extract_and_tokenize_functions_from_csv(csv_file, column_name, pretrain_csv, ft_tosplit_csv)
    #Finalize pretraining dataset
    split_csv_by_percentage(pretrain_csv,output_path1=pretrain_csv)
    #Finalize finetuning and evaluation datasets
    split_csv_by_percentage(ft_tosplit_csv,output_path1=finetune_csv,output_path2=evaluate_csv,pct1=0.25,pct2=0.05,start_pct=0.7)

print('Full dataset masked and tokenized')

df_read = pd.read_csv(pretrain_csv)
df_read['input_ids'] = df_read['input_ids'].apply(ast.literal_eval)
df_read['labels'] = df_read['labels'].apply(ast.literal_eval)
pretrain_set = Dataset.from_pandas(df_read)

df_read = pd.read_csv(finetune_csv)
df_read['input_ids'] = df_read['input_ids'].apply(ast.literal_eval)
df_read['labels'] = df_read['labels'].apply(ast.literal_eval)
finetune_set = Dataset.from_pandas(df_read)

df_read = pd.read_csv(evaluate_csv)
df_read['input_ids'] = df_read['input_ids'].apply(ast.literal_eval)
df_read['labels'] = df_read['labels'].apply(ast.literal_eval)
eval_set = Dataset.from_pandas(df_read)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
) #Modify to our args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Training device:'+str(device))

model_trainer1 = Trainer(
    model=model_base,
    args=training_args,
    train_dataset=pretrain_set,
    data_collator=data_collator
)
model1 = model_trainer1.train()

print('Pretraining step complete')

model_trainer2 = Trainer(model1, train_dataset=finetune_set, eval_dataset=eval_set)
eval1 = model_trainer2.evaluate()
model2 = model_trainer2.train()

print('Finetuning step complete')

model_final_eval = Trainer(model2, train_dataset=finetune_set, eval_dataset=eval_set)
eval2 = model_final_eval.evaluate()