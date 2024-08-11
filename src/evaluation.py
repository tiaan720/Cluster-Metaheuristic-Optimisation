from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np
import logging


class Evaluation:
    """
    Class for evaluating clustering results using various metrics.
    """

    logger = logging.getLogger("Evaluation")

    @staticmethod
    def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the Dunn Index for the given data and cluster labels.

        Args:
            X (np.ndarray): The dataset, where rows are samples and columns are features.
            labels (np.ndarray): Cluster labels for each sample.

        Returns:
            float: The Dunn Index value.
        """
        Evaluation.logger.info("Calculating Dunn Index.")
        distances = pdist(X)
        dist_matrix = squareform(distances)

        unique_labels = np.unique(labels)
        min_inter_cluster_distance = np.inf
        max_intra_cluster_distance = 0

        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                max_intra_cluster_distance = max(
                    max_intra_cluster_distance, np.max(pdist(cluster_points))
                )

            for other_label in unique_labels:
                if label != other_label:
                    other_cluster_points = X[labels == other_label]
                    inter_cluster_distance = np.min(
                        cdist(cluster_points, other_cluster_points)
                    )
                    min_inter_cluster_distance = min(
                        min_inter_cluster_distance, inter_cluster_distance
                    )

        dunn_index = (
            min_inter_cluster_distance / max_intra_cluster_distance
            if max_intra_cluster_distance > 0
            else 0
        )
        Evaluation.logger.info(f"Dunn Index calculated: {dunn_index:.4f}")
        return dunn_index

    @staticmethod
    def wcss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Calculate the Within-Cluster Sum of Squares (WCSS) for the given data and cluster labels.

        Args:
            X (np.ndarray): The dataset, where rows are samples and columns are features.
            labels (np.ndarray): Cluster labels for each sample.
            centroids (np.ndarray): Cluster centroids.

        Returns:
            float: The WCSS value.
        """
        Evaluation.logger.info("Calculating WCSS.")
        wcss = 0.0
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_points = X[labels == label]
            centroid = centroids[label]
            wcss += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
        Evaluation.logger.info(f"WCSS calculated: {wcss:.4f}")
        return wcss

    @staticmethod
    def evaluate(
        X: np.ndarray, labels: np.ndarray, centroids: np.ndarray = None
    ) -> dict:
        """
        Evaluate the clustering results using various metrics.

        Args:
            X (np.ndarray): The dataset, where rows are samples and columns are features.
            labels (np.ndarray): Cluster labels for each sample.
            centroids (np.ndarray, optional): Cluster centroids. Required for WCSS calculation.

        Returns:
            dict: A dictionary containing various clustering evaluation metrics.
        """
        Evaluation.logger.info("Evaluating clustering results.")
        evaluation_results = {
            "silhouette_score": silhouette_score(X, labels),
            "calinski_harabasz_index": calinski_harabasz_score(X, labels),
            "davies_bouldin_index": davies_bouldin_score(X, labels),
            "dunn_index": Evaluation.dunn_index(X, labels),
        }

        if centroids is not None:
            evaluation_results["wcss"] = Evaluation.wcss(X, labels, centroids)
        else:
            Evaluation.logger.warning(
                "WCSS calculation skipped due to missing centroids."
            )

        return evaluation_results
