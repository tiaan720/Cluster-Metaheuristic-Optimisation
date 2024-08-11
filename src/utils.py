import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from kneed import KneeLocator


def determine_optimal_clusters(X: np.ndarray, max_k: int = 10) -> int:
    """
    Determine the optimal number of clusters using the elbow method.

    Args:
        X (np.ndarray): The dataset, where rows are samples and columns are features.
        max_k (int): The maximum number of clusters to consider.

    Returns:
        int: The optimal number of clusters.
    """
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Automatically determine the elbow point using the KneeLocator
    knee_locator = KneeLocator(
        range(1, max_k + 1), wcss, curve="convex", direction="decreasing"
    )
    elbow_point = knee_locator.knee

    if elbow_point is None:
        raise ValueError("KneeLocator could not find an optimal number of clusters.")

    return elbow_point
