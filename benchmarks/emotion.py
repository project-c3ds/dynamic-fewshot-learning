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

data_name = 'emotion'

# Load the data
if not os.path.exists(f'benchmarks/data/{data_name}/fewshot.parquet'):
    df_fewshot = pd.read_csv(f'benchmarks/data/{data_name}/fewshot.csv')
else:
    df_fewshot = pd.read_parquet(f'benchmarks/data/{data_name}/fewshot.parquet')

df_test = pd.read_csv(f'benchmarks/data/{data_name}/test.csv')

# Generate embeddings for the few-shot examples
if not 'embeddings' in df_fewshot.columns:
    df_fewshot['embeddings'] = df_fewshot['text'].progress_apply(generate_embeddings)
    
    df_fewshot.to_parquet(f'benchmarks/data/{data_name}/fewshot.parquet')
else:
    print("Embeddings already generated.")




################################################### Prompt designing ###################################################
prompt_template = f"""
Context:
<text>

If a mental health professional saw the above context, what emotion would they categorize it to be (using the following six basic emotions according to Paul Ekman)?
A: Fear
B: Anger
C: Joy
D: Sadness
E: Love
F: Surprise

Please utilize the examples below as valuable references for guidance and inspiration. Note that they have been selected using semantic search, similar to the context provided above. Take your time to analyze them thoughtfully, as the context may not explicitly indicate the intended emotion. Avoid rushing and reflect on the examples to provide a comprehensive and accurate response.
<examples>

Constraint: Your response should either be "Fear", "Anger", "Joy", "Sadness", "Love", or "Surprise". Don't elaborate on your answer.

"""

prompt_template = """
Imagine three different experts are answering this question. They will brainstorm the answer step by step, reasoning carefully and taking all facts into consideration. All experts will write down 1 step of their thinking, then share it with the group. They will each critique their response, and then all the responses of others. They will check their answer based on mental health knowledge and emotional psychology principles. Then all experts will go on to the next step and write down this step of their thinking. They will keep going through steps until they reach their conclusion, taking into account the thoughts of the other experts. If at any time they realize that there is a flaw in their logic, they will backtrack to where that flaw occurred. If any expert realizes they're wrong at any point, then they acknowledge this and start another train of thought. Each expert will assign a likelihood of their current assertion being correct. Continue until the experts agree on the single most likely emotion.

Context:
<text>

Examples:
<examples>

Available Emotion Classes:
A: Fear
B: Anger
C: Joy
D: Sadness
E: Love
F: Surprise

Think through each step logically.

Expert 1:
Step 1: Expert 1 will analyze the given context, consider the examples, and assess the potential emotion conveyed.
Critique 1: Expert 1 will review their response for any biases or hasty judgments.
Step 2: Expert 1 will refer to their mental health knowledge and emotional psychology principles to choose the most likely emotion class from the available options.
Critique 2: Expert 1 will assess whether their choice aligns with the provided examples and the context.

Expert 2:
Step 1: Expert 2 will independently evaluate the context, study the examples, and consider possible emotions evoked.
Critique 1: Expert 2 will scrutinize their initial assessment and look for any emotional nuances they might have missed.
Step 2: Expert 2 will utilize their understanding of emotional psychology and the provided examples to narrow down the most appropriate emotion class.
Critique 2: Expert 2 will review whether their chosen emotion resonates with the context and the examples provided.

Expert 3:
Step 1: Expert 3 will analyze the context in light of their mental health expertise, study the examples, and consider the emotional undertones.
Critique 1: Expert 3 will critically examine their initial emotional categorization for potential errors or oversights.
Step 2: Expert 3 will draw upon their knowledge of emotional triggers, patterns, and the provided examples to select the emotion class that best fits the context.
Critique 2: Expert 3 will evaluate the alignment between their chosen emotion class and the context, ensuring consistency.

Final Conclusion:
The experts will come together, share their conclusions, and discuss any discrepancies in their reasoning.
They will collaboratively evaluate the likelihood of each emotion class based on their collective expertise and the provided examples.
If any contradictions arise, they will revisit their individual steps to identify any logical flaws.
Through careful consideration and consensus-building, the experts will agree on the single most likely emotion class that a mental health professional would categorize the given context to be.

Constraint: Your response should either be "Fear", "Anger", "Joy", "Sadness", "Love", or "Surprise". Don't elaborate on your answer.
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

notes = '04'
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