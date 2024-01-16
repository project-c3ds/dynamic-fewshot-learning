from isca.icl import get_static_fewshot_examples, get_dynamic_fewshot_examples

instruction_template = ""
prompt_template = ''
def get_zeroshot_prompt(instruction_template=instruction_template):
    """
    Returns the zeroshot prompt template.

    Returns:
        str: The zeroshot prompt template.

    Raises:
        ValueError: If the zeroshot prompt template is an empty string.
    """
    if instruction_template == "":
        raise ValueError("instrction_template is empty. Please write a prompt template and assign it to the instruction_template variable.")
    return instruction_template

def get_fewshot_prompt(examples, instruction_template=instruction_template):
    """
    Returns the fewshot prompt template.

    Args:
        text (str): The text to classify.
        examples (str): The fewshot examples. You should use the get_static_fewshot_examples or get_dynamic_fewshot_examples functions to generate this.
        instruction_template (str): The instruction prompt template. defaults to ''.

    Returns:
        str: The fewshot prompt template.

    Raises:
        ValueError: If the fewshot prompt template or examples are empty strings.
    """
    if instruction_template == "":
        raise ValueError("instrction_template is empty. Please write a prompt template and assign it to the instruction_template variable.")
    elif examples == "":
        raise ValueError("Fewshot examples are empty. You should use the get_static_fewshot_examples or get_dynamic_fewshot_examples functions to generate this")
    else:
        instruction_fewshot_prompt = instruction_template.replace('Text: <text>', f'')
        instruction_fewshot_prompt = f"""
{instruction_template}

Here are some examples for your reference:

{examples}

Text: <text>
"""
    return instruction_fewshot_prompt

def get_all_prompts(text, label,  prompt_template, df_fewshot ,k=10, use_kdtree=False, text_label='TEXT', label_name = 'CATEGORY', groupby_sample=False):

    """

    Get all prompts for zeroshot, static fewshot and dynamic fewshot.

    Args:
        text (str): The text to classify.
        label (str): The label to classify the text into.
        df_fewshot (pandas.DataFrame): DataFrame containing text and classification columns.
        prompt_template (str): The template to use for the prompt. Should contain the string '<text>' to indicate where the text should be inserted.
        k (int): Number of examples to return for each classification. Defaults to 10.
        use_kdtree (bool): Whether to use a k-d tree for the dynamic few-shot examples. Defaults to False.
        text_label (str): The label to use for the text in the prompt. Defaults to 'TEXT'.
        label_name (str): The label to use for the classification in the prompt. Defaults to 'CATEGORY'.
    """

    static_examples, static_prop = get_static_fewshot_examples(label = label,df_fewshot=df_fewshot, k=k, text_label=text_label, label_name=label_name, groupby_sample=groupby_sample)
    dynamic_examples, dynamic_prop = get_dynamic_fewshot_examples(df_fewshot=df_fewshot, text_to_classify=text, label=label, k=k, use_kdtree=use_kdtree, text_label=text_label, label_name=label_name, groupby_sample=groupby_sample)

    zeroshot_prompt = prompt_template.replace('<text>', f'{text}').replace('EXAMPLES: <examples>', '')\
                        .replace("Please utilize the following examples as references for guidance and inspiration. being aware that the context above might not explicitly indicate the intended emotion. Sometimes, the presence of word may not reflect the overall sentiment.", '')\
                        .replace('<examples>', '').replace('Examples:','')\
                        .replace('Please utilize the examples below as valuable references for guidance and inspiration. Note that they have been selected using semantic search, similar to the context provided above. Take your time to analyze them thoughtfully. Avoid rushing and reflect on the examples to provide a comprehensive and accurate response.','')\
                        .replace("Kindly utilize the following examples as valuable references to guide and inspire you. These examples have been carefully selected using semantic search, similar to the context you provided. Take ample time to analyze them thoughtfully, as the context may not explicitly indicate the intended emotion, avoiding any haste, and reflect on the examples to provide a comprehensive and accurate response.",'')\
                        .replace("Kindly utilize the following examples as valuable references to guide and inspire you. These examples have been carefully selected using semantic search, similar to the context you provided. Take ample time to analyze them thoughtfully, avoiding any haste, and reflect on the examples to provide a comprehensive and accurate response:",'')\
                        .replace("Take advantage of the following examples, thoughtfully chosen through semantic search based on the context provided, to guide and inspire you. Dedicate sufficient time to analyze them carefully, aiming for a comprehensive and accurate response:",'')\
                        .replace("Please utilize the examples below as valuable references for guidance and inspiration. Note that they have been selected using semantic search, similar to the Tweet provided above. Take your time to analyze them thoughtfully, as the tweet may not explicitly indicate the intended emotion. Avoid rushing and reflect on the examples to provide a comprehensive and accurate response.",'')
    static_fewshot_prompt = prompt_template.replace('<examples>', f'\n{static_examples}').replace('<text>', f'{text}')
    dynamic_fewshot_prompt = prompt_template.replace('<examples>', f'\n{dynamic_examples}').replace('<text>', f'{text}')

    return zeroshot_prompt, static_fewshot_prompt, dynamic_fewshot_prompt, static_prop, dynamic_prop