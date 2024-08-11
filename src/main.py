import argparse
import logging
import os
import pandas as pd
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import (
    StudentDataset,
    WineDataset,
    IrisDataset,
    BreastCancerDataset,
)
from src.optimization_algorithm import (
    PSOClustering,
    GeneticAlgorithmClustering,
    SimulatedAnnealingClustering,
    TabuSearchClustering,
)
from src.clustering import KMeansClustering
from src.evaluation import Evaluation
from src.visualization import Visualization
from utils import determine_optimal_clusters  # Import the utility function
import numpy as np


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def save_results(results, folder):
    os.makedirs(folder, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_filename = os.path.join(folder, "clustering_evaluation_results.csv")
    results_df.to_csv(results_filename, index=False)
    logging.info(f"Clustering evaluation results saved as {results_filename}.")


def get_dataset(dataset_name):
    if dataset_name == "students":
        return StudentDataset()
    elif dataset_name == "wine":
        return WineDataset()
    elif dataset_name == "iris":
        return IrisDataset()
    elif dataset_name == "breast_cancer":
        return BreastCancerDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_optimization_algorithm(algorithm_name, n_clusters, **kwargs):
    if algorithm_name == "PSO":
        n_particles = kwargs.get("n_particles", 100)
        max_iter = kwargs.get("max_iter", 1000)
        return PSOClustering(
            n_clusters=n_clusters, n_particles=n_particles, max_iter=max_iter
        )
    elif algorithm_name == "GA":
        n_population = kwargs.get("n_population", 500)
        max_iter = kwargs.get("max_iter", 5000)
        mutation_rate = kwargs.get("mutation_rate", 0.1)
        return GeneticAlgorithmClustering(
            n_clusters=n_clusters,
            n_population=n_population,
            max_iter=max_iter,
            mutation_rate=mutation_rate,
        )
    elif algorithm_name == "SA":
        initial_temp = kwargs.get("initial_temp", 10000)
        final_temp = kwargs.get("final_temp", 0.0001)
        alpha = kwargs.get("alpha", 0.99)
        max_iter = kwargs.get("max_iter", 5000)
        return SimulatedAnnealingClustering(
            n_clusters=n_clusters,
            initial_temp=initial_temp,
            final_temp=final_temp,
            alpha=alpha,
            max_iter=max_iter,
        )
    elif algorithm_name == "TS":
        max_iter = kwargs.get("max_iter", 5000)
        tabu_tenure = kwargs.get("tabu_tenure", 50)
        return TabuSearchClustering(
            n_clusters=n_clusters, max_iter=max_iter, tabu_tenure=tabu_tenure
        )
    else:
        raise ValueError(f"Unknown optimization algorithm: {algorithm_name}")


def main(dataset_name, optimization_algorithm, n_clusters=None, **kwargs):
    setup_logging()
    logger = logging.getLogger("Main")

    logger.info("Starting clustering analysis.")

    dataset = get_dataset(dataset_name)

    logger.info("Preprocessing the dataset.")
    X_scaled = dataset.preprocess()

    # Determine the optimal number of clusters if not provided
    if n_clusters is None:
        logger.info(
            "Determining the optimal number of clusters using the elbow method."
        )
        n_clusters = determine_optimal_clusters(X_scaled)

    # Folder for storing results
    base_folder = "cluster_data"
    algorithm_folder = os.path.join(
        base_folder, f"{dataset_name}_{optimization_algorithm}_Clustering_Results"
    )

    # Run K-means clustering
    logger.info("Running K-Means clustering.")
    kmeans = KMeansClustering(n_clusters=n_clusters)
    kmeans_labels, kmeans_wcss = kmeans.fit_predict(X_scaled)
    kmeans_eval = Evaluation.evaluate(X_scaled, kmeans_labels)
    logger.info("K-means Clustering Results:")
    kmeans_results = {
        "Algorithm": "KMeans",
        "WCSS": kmeans_wcss,
        "Silhouette Score": kmeans_eval["silhouette_score"],
        "Calinski Harabasz Index": kmeans_eval["calinski_harabasz_index"],
        "Davies Bouldin Index": kmeans_eval["davies_bouldin_index"],
        "Dunn Index": kmeans_eval["dunn_index"],
    }
    logger.info(kmeans_results)

    # Run optimization algorithm clustering
    logger.info(f"Running {optimization_algorithm} clustering.")
    optimizer = get_optimization_algorithm(
        optimization_algorithm, n_clusters=n_clusters, **kwargs
    )
    opt_labels, opt_results = optimizer.run(X_scaled)
    logger.info(opt_results)

    # Save and visualize the results
    save_results([kmeans_results, opt_results], algorithm_folder)

    logger.info("Visualizing K-Means clustering results.")
    Visualization.plot_clusters(
        X_scaled,
        kmeans_labels,
        "K-means Clustering",
        dataset_name,
        algorithm_folder,
        "KMeans",
    )
    Visualization.plot_clusters(
        X_scaled,
        opt_labels,
        f"{optimization_algorithm} Clustering",
        dataset_name,
        algorithm_folder,
        optimization_algorithm,
    )

    # Visualize convergence and diversity data if available
    if "Convergence Data" in opt_results and "Diversity Data" in opt_results:
        Visualization.plot_convergence(
            opt_results["Convergence Data"],
            dataset_name,
            algorithm_folder,
            optimization_algorithm,
        )
        Visualization.plot_diversity(
            opt_results["Diversity Data"],
            dataset_name,
            algorithm_folder,
            optimization_algorithm,
        )

    logger.info("Clustering analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering analysis.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["students", "wine", "iris", "breast_cancer"],
        help="Dataset to use for clustering.",
    )
    parser.add_argument(
        "--optimization",
        type=str,
        required=True,
        choices=["PSO", "GA", "SA", "TS"],
        help="Optimization algorithm to use for clustering.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters to form. If not provided, the elbow method will be used.",
    )
    parser.add_argument(
        "--n_particles",
        type=int,
        default=500,
        help="Number of particles (for PSO).",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=500,
        help="Maximum number of iterations.",
    )
    parser.add_argument(
        "--n_population",
        type=int,
        default=500,
        help="Population size (for GA).",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.1,
        help="Mutation rate (for GA).",
    )
    parser.add_argument(
        "--initial_temp",
        type=float,
        default=100,
        help="Initial temperature (for SA).",
    )
    parser.add_argument(
        "--final_temp",
        type=float,
        default=0.1,
        help="Final temperature (for SA).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="Alpha (cooling rate) (for SA).",
    )
    parser.add_argument(
        "--tabu_tenure",
        type=int,
        default=10,
        help="Tabu tenure (for TS).",
    )
    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        optimization_algorithm=args.optimization,
        n_clusters=args.n_clusters,
        n_particles=args.n_particles,
        max_iter=args.max_iter,
        n_population=args.n_population,
        mutation_rate=args.mutation_rate,
        initial_temp=args.initial_temp,
        final_temp=args.final_temp,
        alpha=args.alpha,
        tabu_tenure=args.tabu_tenure,
    )
