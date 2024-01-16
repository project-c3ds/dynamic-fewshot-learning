import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')

def generate_embeddings(text, model_name = "fasttext"):
    """
    Generate embeddings for the given text using the given model.
    Args:
        text (str): The text to generate embeddings for.
        model_name (str): The name of the model to use. Defaults to "fasttext".
    Returns:
        numpy.ndarray: The generated embeddings.
    """
    if model_name == "fasttext":

        return ft.get_sentence_vector(text)

    else:
        raise ValueError("Invalid model name. Must be one of 'fasttext'.")

    
