from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import os


class Visualization:
    logger = logging.getLogger("Visualization")

    @staticmethod
    def plot_clusters(X, labels, title, dataset_name, algorithm_folder, plot_name):
        Visualization.logger.info(
            f"Creating t-SNE plot for {title} on {dataset_name} dataset."
        )
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        plt.figure()
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="viridis")
        plt.title(f"{title} ({dataset_name} Dataset)")

        # Save the plot in the appropriate folder
        os.makedirs(algorithm_folder, exist_ok=True)
        plot_filename = os.path.join(
            algorithm_folder, f"{plot_name}_{dataset_name}_clustering_plot.png"
        )
        plt.savefig(plot_filename)
        Visualization.logger.info(f"Plot saved as {plot_filename}.")
        plt.close()

    @staticmethod
    def plot_convergence(
        convergence_data, dataset_name, algorithm_folder, algorithm_name
    ):
        Visualization.logger.info(
            f"Creating convergence plot for {algorithm_name} on {dataset_name} dataset."
        )
        plt.figure(figsize=(12, 6))
        plt.plot(convergence_data, label="Calinski-Harabasz Score")
        plt.xlabel("Iterations")
        plt.ylabel("Calinski-Harabasz Score")
        plt.title(
            f"Convergence of {algorithm_name} Clustering Algorithm ({dataset_name} Dataset)"
        )
        plt.legend()

        plot_filename = os.path.join(
            algorithm_folder, f"{algorithm_name}_{dataset_name}_convergence_plot.png"
        )
        plt.savefig(plot_filename)
        Visualization.logger.info(f"Convergence plot saved as {plot_filename}.")
        plt.close()

    @staticmethod
    def plot_diversity(diversity_data, dataset_name, algorithm_folder, algorithm_name):
        Visualization.logger.info(
            f"Creating diversity plot for {algorithm_name} on {dataset_name} dataset."
        )
        plt.figure(figsize=(12, 6))
        plt.plot(diversity_data, label="Diversity")
        plt.xlabel("Iterations")
        plt.ylabel("Diversity (Standard Deviation)")
        plt.title(
            f"Diversity of Population in {algorithm_name} Clustering Algorithm ({dataset_name} Dataset)"
        )
        plt.legend()

        plot_filename = os.path.join(
            algorithm_folder, f"{algorithm_name}_{dataset_name}_diversity_plot.png"
        )
        plt.savefig(plot_filename)
        Visualization.logger.info(f"Diversity plot saved as {plot_filename}.")
        plt.close()
