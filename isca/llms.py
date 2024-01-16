from openai import OpenAI
import requests
from requests.exceptions import HTTPError
import backoff
from openai.error import RateLimitError, InvalidRequestError
import os

api_key = 'sk-'
client = OpenAI(api_key=api_key)

@backoff.on_exception(backoff.expo, (RateLimitError,InvalidRequestError), max_tries=8)
def get_openai_inference(prompt: str, model_name: str = 'gpt-4', temperature: float = 0.0, max_tokens: int = 20) -> str:
    """
    Use the OpenAI API to generate a response to the given prompt.

    Args:
        prompt (str): The prompt to use as input to the model.
        model_name (str): The name of the OpenAI model to use. Defaults to 'gpt-3.5-turbo'. You can use any of the models listed at https://www.openai.com/docs/api-reference/
        temperature (float): A value controlling the randomness of the model's output. 
            Lower values produce more conservative predictions, while higher values produce more creative ones. 
            Defaults to 0.0.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 20.

    Returns:
        str: The generated response to the given prompt.
    """
    if model_name == 'gpt-3.5-turbo' or model_name == 'gpt-4':
        completion = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message['content']
    else:
        completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens)
        return completion.choices[0].text


def get_community_model_api_inference(message: str, model_name: str = 'vicuna', temperature: float = 0.0, max_tokens: int = 20) -> str:
    """
    Use the Vicuna API to generate a response to the given prompt.

    Args:
        message (str): The prompt to use as input to the model.
        modelname (str): The name of the model to use. Defaults to 'wizardvicuna'. You can use any of the models listed at
        temperature (float): A value controlling the randomness of the model's output.
            Lower values produce more conservative predictions, while higher values produce more creative ones.
            Defaults to 0.0.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 20.
    """
    url = "http://IP-Address:8000/v1/chat/completions" # Set this up using Vllm or FastChat
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']
