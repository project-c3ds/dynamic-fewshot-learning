from datetime import datetime
import pandas as pd
import sqlite3

import sys
# Append the root directory of the project to the path using os independent code
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


import os
current_dir = os.getcwd()
base_dir = os.path.dirname(current_dir)
data_dir = os.path.join(base_dir, 'benchmarks/data/predictions')

from isca.llms import get_openai_inference, get_community_model_api_inference
from isca.prompts import get_fewshot_prompt, get_zeroshot_prompt, instruction_template, get_all_prompts
from isca.icl import get_static_fewshot_examples, get_dynamic_fewshot_examples
from isca.embeddings import generate_embeddings


def save_result_to_db(tmp_dict, table_name, con):
    """
    Save the result to the database.
    Args:
        tmp_dict (dict): The dictionary containing the result.
        table_name (str): The name of the table to save the result to.

    Returns:
        None
    """

    timestamp = str(datetime.now())
    tmp_dict['timestamp'] = timestamp
    tmp_df = pd.DataFrame(tmp_dict, index=[0])
    
    tmp_df.to_sql(table_name, if_exists='append', index=False, con=con)
    
    return None

def get_all_predictions_openai(text, label,zeroshot_prompt, static_fewshot_prompt, dynamic_fewshot_prompt, static_prop, dynamic_prop ,max_tokens=10, model_name='gpt-3.5-turbo', temperature=0.0, select_technique='all'):

    """
    Get predictions from the OpenAI API. 

    Args:
        text (str): The text to classify.
        label (str): The label to classify the text into.
        df_fewshot (pandas.DataFrame): DataFrame containing text and classification columns.
        static_examples (list): List of examples to use for the few-shot classification.
        instruction_template (str): The template to use for the prompt. Should contain the string '<text>' to indicate where the text should be inserted.
        n (int): Number of examples to return for each classification. Defaults to 10.
        use_kdtree (bool): Whether to use a k-d tree for the dynamic few-shot examples. Defaults to False.
        dynamic_fewshot_prompt (str): The prompt to use for the dynamic few-shot examples. If None, the prompt is generated using the `get_fewshot_prompt` function. Defaults to None.

    returns:
        tmp_dict (dict): The dictionary containing the result.

    """
    if select_technique == 'all':
        zeroshot_response = get_openai_inference(prompt=zeroshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)
        static_fewshot_response = get_openai_inference(prompt=static_fewshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)
        dynamic_fewshot_response = get_openai_inference(prompt=dynamic_fewshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)
    elif select_technique == 'zeroshot':
        zeroshot_response = get_openai_inference(prompt=zeroshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)
        static_fewshot_response = 'skipped'
        dynamic_fewshot_response = 'skipped'
    elif select_technique == 'static':
        zeroshot_response = 'skipped'
        static_fewshot_response = get_openai_inference(prompt=static_fewshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)
        dynamic_fewshot_response = 'skipped'
    elif select_technique == 'dynamic':
        zeroshot_response = 'skipped'
        static_fewshot_response = 'skipped'
        dynamic_fewshot_response = get_openai_inference(prompt=dynamic_fewshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)

    tmp_dict = {}
    tmp_dict['zeroshot_prompt'] = zeroshot_prompt
    tmp_dict['static_fewshot_prompt'] = static_fewshot_prompt
    tmp_dict['dynamic_fewshot_prompt'] = dynamic_fewshot_prompt
    tmp_dict['text'] = text
    tmp_dict['static_prop'] = static_prop
    tmp_dict['dynamic_prop'] = dynamic_prop
    tmp_dict['zeroshot_response'] = zeroshot_response
    tmp_dict['static_fewshot_response'] = static_fewshot_response
    tmp_dict['dynamic_fewshot_response'] = dynamic_fewshot_response
    tmp_dict['label'] = label
    
    return tmp_dict



def get_all_predictions_community_api(text, label,zeroshot_prompt, static_fewshot_prompt, dynamic_fewshot_prompt, static_prop, dynamic_prop ,model_name='wizardvicuna', max_tokens=30, temperature=0.0, select_technique='all'):

    """
    Get predictions from the community model API.
    Args:
        text (str): The text to classify.
        label (str): The label to classify the text into.
        df_fewshot (pandas.DataFrame): DataFrame containing text and classification columns.
        static_examples (list): List of examples to use for the few-shot classification.
        instruction_template (str): The template to use for the prompt. Should contain the string '<text>' to indicate where the text should be inserted.
        n (int): Number of examples to return for each classification. Defaults to 10.
        model_name (str): The name of the model to use for the community model API. Defaults to 'wizardvicuna'.
        max_tokens (int): The maximum number of tokens to use for the prompt. Defaults to 30.
        use_kdtree (bool): Whether to use a k-d tree for the dynamic few-shot examples. Defaults to False.
        fewshot_prompt (str): The prompt to use for the dynamic few-shot examples. If None, the prompt is generated using the `get_fewshot_prompt` function. Defaults to None.

    """

    if select_technique == 'all':
        zeroshot_response = get_community_model_api_inference(zeroshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)
        static_fewshot_response = get_community_model_api_inference(static_fewshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)
        dynamic_fewshot_response = get_community_model_api_inference(dynamic_fewshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)
    elif select_technique == 'zeroshot':
        zeroshot_response = get_community_model_api_inference(zeroshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)
        static_fewshot_response = 'skipped'
        dynamic_fewshot_response = 'skipped'
    elif select_technique == 'static':
        zeroshot_response = 'skipped'
        static_fewshot_response = get_community_model_api_inference(static_fewshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)
        dynamic_fewshot_response = 'skipped'
    elif select_technique == 'dynamic':
        zeroshot_response = 'skipped'
        static_fewshot_response = 'skipped'
        dynamic_fewshot_response = get_community_model_api_inference(dynamic_fewshot_prompt, model_name = model_name, max_tokens=max_tokens, temperature=temperature)

    tmp_dict = {}
    tmp_dict['zeroshot_prompt'] = zeroshot_prompt
    tmp_dict['static_fewshot_prompt'] = static_fewshot_prompt
    tmp_dict['dynamic_fewshot_prompt'] = dynamic_fewshot_prompt
    tmp_dict['text'] = text
    tmp_dict['static_prop'] = static_prop
    tmp_dict['dynamic_prop'] = dynamic_prop
    tmp_dict['zeroshot_response'] = zeroshot_response
    tmp_dict['static_fewshot_response'] = static_fewshot_response
    tmp_dict['dynamic_fewshot_response'] = dynamic_fewshot_response
    tmp_dict['label'] = label
    
        

    return tmp_dict
