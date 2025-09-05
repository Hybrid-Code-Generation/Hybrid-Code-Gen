import pandas as pd
import numpy as np

def search_method(
    method_name: str,
    class_name: str = "",
    parameters: str = "",
    return_type: str = "",
):
    """
    Search for a method in the knowledge graph and retrieve its context.

    Args:
        method_name: Name of the method to search for.
        class_name: Name of the class containing the method (optional).
        parameters: Method parameters as a string (optional).
        return_type: Return type of the method (optional).

    Returns:
        The entire row for that method in methods.csv
    """
    csv_path = "./data/methods.csv"
    df = pd.read_csv(csv_path)

    # Filter the DataFrame based on the provided criteria
    filtered_df = df[df['Method Name'] == method_name]
    if class_name:
        filtered_df = filtered_df[filtered_df['Class'] == class_name]
    if parameters:
        filtered_df = filtered_df[filtered_df['Parameters'] == parameters]
    if return_type:
        filtered_df = filtered_df[filtered_df['Return Type'] == return_type]

    # If no matching method is found, return None
    if filtered_df.empty:
        return None

    # Return the first matching row as a dictionary
    return filtered_df.iloc[0].to_dict()



def search_method_csv_weighted(
    user_query: str,
    method_name: str,
    class_name: str = "",
    parameters: str = "",
    return_type: str = "",
):
    """
    Search for a method in the knowledge graph and retrieve its context.

    Args:
        method_name: Name of the method to search for.
        class_name: Name of the class containing the method (optional).
        parameters: Method parameters as a string (optional).
        return_type: Return type of the method (optional).

    Returns:
        The entire row for that method in methods.csv
    """
    csv_path = "./data/methods.csv"
    df = pd.read_csv(csv_path)

    # Filter the DataFrame based on the provided criteria
    filtered_df = df[df['Method Name'] == method_name]
    if class_name:
        filtered_df = filtered_df[filtered_df['Class'] == class_name]
    if parameters:
        filtered_df = filtered_df[filtered_df['Parameters'] == parameters]
    if return_type:
        filtered_df = filtered_df[filtered_df['Return Type'] == return_type]

    # If no matching method is found, return None
    if filtered_df.empty:
        return None

    # Import and initialize CodeSearcher
    from processor.query_search_OpenAI import CodeSearcher
    searcher = CodeSearcher()
    # Only initialize if not already done (avoid repeated downloads)
    if searcher.client is None or searcher.df is None or searcher.index is None:
        searcher.initialize()

    # Use the method body for embedding and similarity calculation
    row = filtered_df.iloc[0]
    method_body = row.get('Body', '') if 'Body' in row else ''
    if not method_body:
        # Fallback to signature if body is missing
        method_body = f"{row['Class']} {row['Method Name']} {row['Parameters']} {row['Return Type']}"

    # Get query embedding (still use cleaned query text)
    query_embedding = searcher.get_code_embedding(user_query)
    # Get method embedding using the body
    method_embedding = searcher.get_code_embedding(method_body)

    q_emb = query_embedding.flatten()
    m_emb = method_embedding.flatten()
    if np.linalg.norm(q_emb) == 0 or np.linalg.norm(m_emb) == 0:
        dynamic_similarity = 0.0
    else:
        dynamic_similarity = float(np.dot(q_emb, m_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(m_emb)))

    static_weight = filtered_df['Static Weight'] if 'Static Weight' in filtered_df else 0.0

    result = row.to_dict()
    result['similarity_score'] = dynamic_similarity * 0.7 + static_weight * 0.3
    print(f"Similarity of query to method {row['Class']}.{row['Method Name']}({row['Parameters']}): {result['similarity_score']}. With static weight: {static_weight}, dynamic similarity: {dynamic_similarity}")
    return result