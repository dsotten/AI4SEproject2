import loaddata
import extract_if
#import kate's_dataset_creator

from transformers import T5ForConditionalGeneration
from transformers import LlamaTokenizer
from transformers import Trainer

model_base = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")

pretrain_set = #kate's_dataset_creator.pretrain_set()
finetune_set = #kate's_dataset_creator.finetune_set()
eval_set = #kate's_dataset_creator.eval_set()

model_trainer1 = Trainer(model_base, train_dataset=pretrain_set, eval_dataset=eval_set)
model1 = model_trainer1.train

model_trainer2 = Trainer(model1, train_dataset=finetune_set, eval_dataset=eval_set)
eval1 = model_trainer2.evaluate
model2 = model_trainer2.train

model_final_eval = Trainer(model2, train_dataset=finetune_set, eval_dataset=eval_set)
eval2 = model_final_eval.evaluate

text = #kate's_dataset_creator.random_sample()
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate a single sequence
generated_ids = model2.generate(input_ids, max_length=10)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))