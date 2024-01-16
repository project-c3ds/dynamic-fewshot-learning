from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.neighbors import KDTree
import pandas as pd
from .embeddings import generate_embeddings

def get_dynamic_fewshot_examples(text_to_classify, label, df_fewshot,  k=5, use_kdtree=False, text_label='text', label_name = 'category', groupby_sample=False):
    """
    Generates a dynamic set of few-shot examples based on a query using cosine similarity or KDTree for nearest neighbors search.
    The df_fewshot DataFrame must contain an "embeddings" column containing the embeddings for each text and a "label" column containing the label label for each text.

    Args:
        query (str): The query text.
        df_fewshot (pandas.DataFrame): The DataFrame containing the few-shot examples to search in.
        n (int): The number of similar texts to return.
        use_kdtree (bool): Whether to use KDTree for nearest neighbors search. If False, cosine similarity will be used.
        return_prop (bool): Whether to return the proportion of similar texts that belong to the same label as the query.

    Returns:
        str: A string containing the n most similar texts to the query in the format:
             text:
             <text>
             label:
             <label_label>
        float: The proportion of similar texts that belong to the same label as the query.

    Raises:
        ValueError: If the input DataFrame is empty or does not contain an "embeddings" column.
        TypeError: If the input query is not a string or is empty.
        ValueError: You need to pass the true label of the query text to get the proportion of similar texts that belong to the same label.

    """
    query = text_to_classify
    if not isinstance(query, str) or not query:
        raise TypeError("text_to_classify must be a non-empty string.")

    query_embedding = generate_embeddings(query)
    
    if "embeddings" not in df_fewshot.columns:
        raise ValueError("DataFrame should contain an 'embeddings' column.")
    if use_kdtree:
        # Create KDTree from the embeddings in the DataFrame
        embeddings = list(df_fewshot['embeddings'])
        tree = KDTree(embeddings)

        # Query the KDTree to find the nearest neighbors
        distances, indices = tree.query([query_embedding], k=k)
        results_df = df_fewshot.iloc[indices[0]]
    else:
        # Calculate cosine similarity between query embedding and all other embeddings in the DataFrame
        similarities = []
        for embedding in df_fewshot['embeddings']:
            similarity = 1 - cosine(query_embedding, embedding)
            similarities.append(similarity)

        # Add the similarity scores to the DataFrame
        df_fewshot['similarity'] = similarities

        # Sort the DataFrame by similarity score in descending order
        sorted_df = df_fewshot.sort_values('similarity', ascending=False)

        # Get the n most similar texts
        if groupby_sample:
            results_df = sorted_df.groupby('label').head(k)
        else:
            results_df = sorted_df.head(k)

    results_df = results_df.sort_values('label')
    records_list = results_df[['text', 'label']].to_dict('records')
    output = ""
    for index, item in enumerate(records_list):
        output += f"\n{text_label} {index + 1}: {item['text']}\n"
        output += f"{label_name} {index + 1}: {item['label']}\n"
    if not label:
        return output, -1
    else:
        prop = len(results_df[results_df.label == label]) / k * 100
        return output, prop


def get_static_fewshot_examples(label, df_fewshot, k=5, text_label='text', label_name = 'category', groupby_sample=False):
    """
    Returns a string containing n examples of text with their associated classification from a pandas DataFrame.

    Args:
        df_fewshot (pandas.DataFrame): DataFrame containing text and classification columns.
        n (int): Number of examples to return for each classification.
        return_prop (bool): Whether to return the proportion of similar texts that belong to the same label as the query.

    Returns:
        str: A string containing n examples of text with their associated classification, formatted as follows:
             text:
             <text>
             label:
             <label>
        float: The proportion of similar texts that belong to the same label as the query.

    Raises:
        ValueError: If the input DataFrame is empty or does not contain a "text" and "label" columns.
    """
    if df_fewshot.empty or 'label' not in df_fewshot.columns:
        raise ValueError("The input DataFrame must not be empty and must contain a 'label' column.")

    if groupby_sample:
        results_df = df_fewshot.groupby('label').sample(k, random_state=123)
    else:
        results_df = df_fewshot.sample(k, random_state=123)
    records_list = results_df[['text', 'label']].to_dict('records')
    output = ""
    for index, item in enumerate(records_list):
        output += f"\n{text_label} {index + 1}: {item['text']}\n"
        output += f"{label_name} {index + 1}: {item['label']}\n"
    prop = len(results_df[results_df.label == label]) / k * 100
    return output, prop
