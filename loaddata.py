import json
import re

import pandas as pd
from pathlib import Path
pd.set_option('max_colwidth',300)
from pprint import pprint

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast

with open('python/final/jsonl/test/python_test_0.jsonl', 'r') as f:
    sample_file = f.readlines()
sample_file[0]

pprint(json.loads(sample_file[0]))

python_files = sorted(Path('python').glob('**/*.gz'))

print(f'Total number of files: {len(python_files):,}')

columns_long_list = ['repo', 'path', 'url', 'code', 
                     'code_tokens', 'docstring', 'docstring_tokens', 
                     'language', 'partition']

columns_short_list = ['code_tokens', 'docstring_tokens', 
                      'language', 'partition']

def jsonl_list_to_dataframe(file_list, columns=columns_long_list):
    """Load a list of jsonl.gz files into a pandas DataFrame."""
    return pd.concat([pd.read_json(f, 
                                   orient='records', 
                                   compression='gzip',
                                   lines=True)[columns] 
                      for f in file_list], sort=False)

df = jsonl_list_to_dataframe(python_files)

df.head(3)
data = df['code']
data.to_csv('full.csv', index=False)