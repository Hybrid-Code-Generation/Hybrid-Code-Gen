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
    csv_path = "methods.csv"
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
