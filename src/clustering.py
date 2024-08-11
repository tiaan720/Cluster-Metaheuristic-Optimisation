import numpy as np  # Import NumPy
from sklearn.cluster import KMeans
import logging


class KMeansClustering:
    def __init__(self, n_clusters: int):
        """
        Initialize the KMeansClustering class.

        Args:
            n_clusters (int): The number of clusters to form.
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit_predict(self, X: np.ndarray) -> tuple:
        """
        Fit the KMeans model to the data and predict the cluster labels.

        Args:
            X (np.ndarray): The dataset to cluster, where rows are samples and columns are features.

        Returns:
            tuple: A tuple containing the cluster labels and the inertia (WCSS).
        """
        self.logger.info(f"Running K-Means clustering with {self.n_clusters} clusters.")
        labels = self.model.fit_predict(X)
        inertia = self.model.inertia_
        self.logger.info("K-Means clustering complete.")
        return labels, inertia
