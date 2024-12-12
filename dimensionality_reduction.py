import pandas as pd
from sklearn.manifold import Isomap, TSNE


def isomap_reduction(df: pd.DataFrame, reduction_size: int) -> pd.DataFrame:
    """
    Reduces dimensionality of a DataFrame using Isomap.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with features to reduce.
        reduction_size (int): The target number of dimensions.
    
    Returns:
        pd.DataFrame: DataFrame with reduced dimensions.
    """
    
    embedding = Isomap(n_components=reduction_size)
    reduced_data = embedding.fit_transform(df)
    return pd.DataFrame(reduced_data, columns=[f"dim_{i+1}" for i in range(reduction_size)])

def tsne_reduction(df: pd.DataFrame, reduction_size: int, perplexity: int = 30, random_state: int = 42) -> pd.DataFrame:
    """
    Reduces dimensionality of a DataFrame using t-SNE.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with features to reduce.
        reduction_size (int): The target number of dimensions.
        perplexity (int): Perplexity parameter for t-SNE (default=30).
        random_state (int): Random state for reproducibility (default=42).
    
    Returns:
        pd.DataFrame: DataFrame with reduced dimensions.
    """
    tsne = TSNE(n_components=reduction_size, perplexity=perplexity, random_state=random_state)
    reduced_data = tsne.fit_transform(df)
    return pd.DataFrame(reduced_data, columns=[f"dim_{i+1}" for i in range(reduction_size)])



