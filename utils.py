import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.decomposition import PCA

def load_embeddings(file_path):
    """
    Load image embeddings from a pickle file.
    """
    return pd.read_pickle(file_path)

def compute_similarities(query_embedding, embeddings):
    """
    Compute cosine similarities between the query embedding and dataset embeddings.
    """
    return F.cosine_similarity(query_embedding, embeddings)

def find_top_k_similar(query_embedding, embeddings, df, k=5):
    """
    Find the top k most similar images to the query embedding.

    Args:
        query_embedding: The embedding for the query (image, text, or hybrid).
        embeddings: Tensor of all dataset embeddings.
        df: Dataframe containing image file names and embeddings.
        k: Number of top results to retrieve.

    Returns:
        A list of dictionaries with 'file_name' and 'similarity' keys.
    """
    similarities = F.cosine_similarity(query_embedding, embeddings)
    top_k_indices = similarities.topk(k).indices
    results = [
        {
            "file_name": df.iloc[int(i)]["file_name"],  # Convert i to an integer
            "similarity": similarities[i].item(),
        }
        for i in top_k_indices
    ]
    return results


def apply_pca(embeddings, n_components=5):
    """
    Reduce embeddings to the top n principal components using PCA.

    Args:
        embeddings: Tensor of embeddings.
        n_components: Number of PCA components to retain.

    Returns:
        Reduced embeddings and the PCA model.
    """
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings.numpy())
    return reduced_embeddings, pca
