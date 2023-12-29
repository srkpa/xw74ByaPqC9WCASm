import pandas as pd

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def compose_and_return(*functions):
    """
    Composes multiple functions into a single function and returns it.

    Args:
        *functions (Callable): Variable number of functions to be composed.

    Returns:
        Callable: The composed function.

    Examples:
        >>> def add_one(x):
        ...     return x + 1
        ...
        >>> def multiply_by_two(x):
        ...     return x * 2
        ...
        >>> composed = compose_and_return(add_one, multiply_by_two)
        >>> composed(3)
        8"""

    def composed_function(input_value):
        result = input_value
        for func in functions:
            result = func(result)
        return result

    return composed_function


def load_csv_to_dataframe(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.

    Examples:
        >>> df = load_csv_to_dataframe('data.csv')"""

    return pd.read_csv(file_path)


def apply_transformations_to_columns(dataframe, transformations, column_names):
    """
    Apply transformations to specified columns in a DataFrame.

    Args:
        dataframe: The DataFrame to apply transformations to.
        transformations: A list of transformation functions.
        column_names: A list of column names to apply the transformations to.

    Returns:
        A copy of the DataFrame with the specified transformations applied to the specified columns.

    Raises:
        ValueError: If the number of transformations does not match the number of column names or if a column is not found in the DataFrame.
    """
    if len(transformations) != len(column_names):
        raise ValueError(
            "The number of transformations must match the number of column names."
        )

    df_copy = dataframe.copy()

    for transform, col_name in zip(transformations, column_names):
        if col_name not in df_copy.columns:
            raise ValueError(f"Column '{col_name}' not found in the DataFrame.")

        df_copy[col_name] = df_copy[col_name].apply(transform)

    return df_copy


def calculate_cosine_similarities(X, Y):
    """
    Calculate the cosine similarities between two sets of vectors.

    Args:
        X: The first set of vectors.
        Y: The second set of vectors.

    Returns:
        The cosine similarities between the vectors in X and Y.

    """
    return 1 - pairwise_distances(X, Y, metric="cosine")
