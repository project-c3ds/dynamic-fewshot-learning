import sys
sys.path.append('/home/projects/isca/')

from isca.llms import get_openai_inference, get_community_model_api_inference
from isca.prompts import get_fewshot_prompt, get_zeroshot_prompt, instruction_template, get_all_prompts
from isca.icl import get_static_fewshot_examples, get_dynamic_fewshot_examples
from isca.embeddings import generate_embeddings
from isca.utils import save_result_to_db, get_all_predictions_openai, get_all_predictions_community_api

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
data_dir = os.path.join(base_dir, 'benchmarks/data')
db_dir = os.path.join(base_dir, 'benchmarks/data/predictions')
# print(db_dir)
import sqlite3

data_name = 'dialects'

# Load the data
if not os.path.exists(f'benchmarks/data/{data_name}/fewshot.parquet'):
    df_fewshot = pd.read_csv(f'benchmarks/data/{data_name}/fewshot.csv')
else:
    df_fewshot = pd.read_parquet(f'benchmarks/data/{data_name}/fewshot.parquet')

# Generate embeddings for the few-shot examples
if not 'embeddings' in df_fewshot.columns:
    df_fewshot['embeddings'] = df_fewshot['text'].progress_apply(generate_embeddings)
    
    df_fewshot.to_parquet(f'benchmarks/data/{data_name}/fewshot.parquet')
else:
    print("Embeddings already generated.")


df_test = pd.read_csv(f'benchmarks/data/{data_name}/test.csv')



################################################### Prompt designing ###################################################
prompt_template = f"""
Text:
<text>

Which of the following features would a linguist say that the above sentence has?
A: Article Omission (e.g., 'Person I like most is here.')
B: Copula Omission (e.g., 'Everything busy in our life.')
C: Direct Object Pronoun Drop (e.g., 'He didn’t give me.')
D: Extraneous Article (e.g, 'Educated people get a good money.')
E: Focus Itself (e.g, 'I did it in the month of June itself.')
F: Focus Only (e.g, 'I was there yesterday only'.)
G: General Extender "and all" (e.g, 'My parents and siblings and all really enjoy it'.)
H: Habitual Progressive (e.g., 'They are getting H1B visas.')
I: Invariant Tag "isn’t it, no, na" (e.g., 'It’s come from me, no?')
J: Inversion In Embedded Clause (e.g., 'The school called to ask when are you going back.')
K: Lack Of Agreement (e.g., 'He talk to them.')
L: Lack Of Inversion In Wh-questions (e.g., 'What are you doing?')
M: Left Dislocation (e.g., 'My parents, they really enjoy playing board games.')
N: Mass Nouns As Count Nouns (e.g., 'They use proper grammars there.')
O: Non-initial Existential "is / are there" (e.g., 'Every year inflation is there.')
P: Object Fronting (e.g., 'In fifteen years, lot of changes we have seen.')
Q: Prepositional Phrase Fronting With Reduction (e.g., 'First of all, right side we can see a plate.')
R: Preposition Omission (e.g., 'I stayed alone two years.')
S: Resumptive Object Pronoun (e.g., 'Some teachers when I was in school I liked them very much.')
T: Resumptive Subject Pronoun (e.g., 'A person living in Calcutta, which he didn’t know Hindi earlier, when he comes to Delhi he has to learn English.')
U: Stative Progressive (e.g., 'We will be knowing how much the structure is getting deflected.')
V: Topicalized Non-argument Constituent (e.g., 'in the daytime I work for the courier service')

Please utilize the examples below as valuable references for guidance and inspiration. Note that they have been selected using semantic search, similar to the context provided above. Take your time to analyze them thoughtfully. Avoid rushing and reflect on the examples to provide a comprehensive and accurate response.
<examples>

Constraint: Answer with only the option above that is most accurate and nothing else. Your response should be as concise as possible.

"""

#####################################################################################################


# Static Fewshot examples
# number of dynamic fewshot examples
# this can either be wizardvicuna or openai, because the db gets created based on the model name

k=5
text_label = 'Text'
label_name = 'Answer'

table_name = data_name

use_kdtree = False

notes = '1'
groupby_sample = False
select_technique = 'all'


#########################################################################
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description of your script')

# Add arguments
parser.add_argument('--model_name', type=str, help='Name of the model')
parser.add_argument('--temperature', type=float, help='Temperature value', default=0.0)
parser.add_argument('--max_tokens', type=int, help='Maximum number of tokens', default=100)

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
model_name = args.model_name
temperature = args.temperature
max_tokens = args.max_tokens
#########################################################################


con = sqlite3.connect(f'{db_dir}/{model_name}.db')

try:
    df_completed = pd.read_sql_query(f'select * from {table_name} where notes={notes}', con)
    completed_texts = set(df_completed.text.unique())
    # completed_texts = set()
    print(len(completed_texts))
except:
    completed_texts = set()
    print(0)

print('Completed texts: ', len(completed_texts))

df_test = df_test.sample(frac=1, random_state=1).reset_index(drop=True)

# df_test = df_test.groupby('label').sample(10, random_state=1234).sample(frac=1).reset_index(drop=True)

print('Number of texts to be classified: ', len(df_test))

results = []
for index, row in tqdm(df_test.iterrows()):
    text = row['text']
    label = row['label']
    if text in completed_texts:
        continue
    else:
        try:    
            zeroshot_prompt, static_fewshot_prompt, dynamic_fewshot_prompt, static_prop, dynamic_prop = get_all_prompts(text=text, label=label, df_fewshot=df_fewshot, prompt_template=prompt_template, k=k, use_kdtree=use_kdtree, text_label=text_label, label_name = label_name, groupby_sample=groupby_sample)
            if model_name == 'gpt-4' or model_name == 'openai':
                tmp_dict = get_all_predictions_openai(text=text, label=label, zeroshot_prompt=zeroshot_prompt, static_fewshot_prompt=static_fewshot_prompt, dynamic_fewshot_prompt=dynamic_fewshot_prompt, static_prop=static_prop, dynamic_prop=dynamic_prop, model_name=model_name ,max_tokens=max_tokens, temperature=temperature, select_technique=select_technique)
            else:                                               
                tmp_dict = get_all_predictions_community_api(text=text, label=label, zeroshot_prompt=zeroshot_prompt, static_fewshot_prompt=static_fewshot_prompt, dynamic_fewshot_prompt=dynamic_fewshot_prompt,  static_prop=static_prop, dynamic_prop=dynamic_prop, model_name=model_name, max_tokens=max_tokens, temperature=temperature, select_technique=select_technique)
            tmp_dict['model_name'] = model_name
            tmp_dict['k'] = k
            tmp_dict['notes'] = notes
            tmp_df = pd.DataFrame(tmp_dict, index=[0])
            save_result_to_db(tmp_dict = tmp_dict, table_name = table_name, con=con)
        except Exception as e:
            print(e)
            continue