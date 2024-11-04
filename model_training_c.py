import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import loaddata
import extract_and_mask_if_2
#import mask_15pct_tokens
#import mask_if_conditions
import pandas as pd
import ast
import torch

from pathlib import Path
import pickle

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

model_base = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_base, padding="longest")
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

#Tokenize full dataset
dataset_filepath = 'python/final/jsonl/test/python_test_1.jsonl'
# pretrain_csv = 'pretrain.csv'
# finetune_csv = 'finetune.csv'
# evaluate_csv = 'evaluate.csv'
csv_file = 'full.csv'
column_name = 'code'
pretrain_csv = 'pretrain_dataset.csv'
finetune_train_csv = 'finetune_train_dataset.csv'
finetune_eval_csv = 'finetune_eval_dataset.csv'
finetune_test_csv = 'finetune_test_dataset.csv'

# extract_and_tokenize_functions_from_csv(csv_file, column_name, pretrain_output, finetune_train_output, finetune_eval_output, finetune_test_output)


if Path(csv_file).is_file():
    pass
else:
    loaddata.main(dataset_filepath)

if Path(pretrain_csv).is_file() and Path(finetune_train_csv).is_file():
    print('Datasets found, skipping tokenization and masking')
    pass
else:
    extract_and_mask_if_2.extract_and_tokenize_functions_from_csv(csv_file, column_name, pretrain_csv, finetune_train_csv, finetune_eval_csv, finetune_test_csv)

print('Full dataset masked and tokenized')

df_read = pd.read_csv(pretrain_csv)
df_read['input_ids'] = df_read['input_ids'].apply(ast.literal_eval)
df_read['labels'] = df_read['labels'].apply(ast.literal_eval)
pretrain_set = Dataset.from_pandas(df_read)

df_read = pd.read_csv(finetune_train_csv)
df_read['input_ids'] = df_read['input_ids'].apply(ast.literal_eval)
df_read['labels'] = df_read['labels'].apply(ast.literal_eval)
finetune_set = Dataset.from_pandas(df_read)

df_read = pd.read_csv(finetune_eval_csv)
df_read['input_ids'] = df_read['input_ids'].apply(ast.literal_eval)
df_read['labels'] = df_read['labels'].apply(ast.literal_eval)
eval_set = Dataset.from_pandas(df_read)

# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     logging_dir='./logs',
#     logging_steps=10,
# ) #Modify to our args

# Enable memory efficient attention if using a transformer model
model_base.config.use_cache = False  # Disable caching during training

# Define training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    # Reduce batch size
    per_device_train_batch_size=1,  # Reduced from 8
    #per_device_eval_batch_size=1,   # Reduced from 8
    # Enable gradient accumulation
    gradient_accumulation_steps=4,
    # Memory optimizations
    fp16=True,                      # Use mixed precision training
    gradient_checkpointing=True,    # Trade compute for memory
    # Optional: enable memory efficient attention
    optim='adamw_torch',            # Use memory efficient optimizer
    # Logging
    logging_dir='./logs',
    logging_steps=10,
    # Memory management
    max_grad_norm=1.0,              # Clip gradients
    dataloader_pin_memory=False,    # Reduce memory usage
    dataloader_num_workers=0,       # Disable multi-processing to reduce memory overhead
)

# Clear GPU cache before training
torch.cuda.empty_cache()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print('Training device: '+str(device))
# print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

model_trainer1 = Trainer(
    model=model_base,
    args=training_args,
    train_dataset=pretrain_set,
    data_collator=data_collator
)
model1 = model_trainer1.train()
file_path = './trained'
with open(file_path, 'wb') as f:
        pickle.dump(model1, f)

print('Pretraining step complete')

model_trainer2 = Trainer(model1, args=training_args, train_dataset=finetune_set, eval_dataset=eval_set, data_collator=data_collator)
# eval1 = model_trainer2.evaluate()
# print(eval1)
model2 = model_trainer2.train()
print('Finetuning step complete')
model_trainer2.save_model('./finetuned_model')
print('Fine-tuned model saved.')


model_final_eval = Trainer(model2, args=training_args, train_dataset=finetune_set, eval_dataset=eval_set, data_collator=data_collator)
eval2 = model_final_eval.evaluate()
print(eval2)