import numpy as np
import random
import logging
from evaluation import Evaluation
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist  # Import cdist
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from pyswarm import pso
from deap import base, creator, tools, algorithms
from scipy.optimize import dual_annealing
import numpy as np
from sklearn.cluster import DBSCAN
import time
from sklearn.metrics import calinski_harabasz_score


def dbscan_initialization(X, n_clusters):
    """
    Initialize centroids using DBSCAN method.

    Args:
        X (np.ndarray): The dataset, where rows are samples and columns are features.
        n_clusters (int): The desired number of clusters.

    Returns:
        np.ndarray: Initial centroids.
    """
    dbscan = DBSCAN(eps=0.8, min_samples=5).fit(X)
    core_sample_indices = dbscan.core_sample_indices_
    core_samples = X[core_sample_indices]

    # If the number of core samples is greater than the number of clusters, select the first n_clusters core points.
    if len(core_samples) >= n_clusters:
        centroids = core_samples[:n_clusters]
    else:
        # If not enough core points, augment with random points.
        random_centroids = np.random.uniform(
            np.min(X, axis=0),
            np.max(X, axis=0),
            (n_clusters - len(core_samples), X.shape[1]),
        )
        centroids = np.vstack((core_samples, random_centroids))

    return centroids


class Particle:
    def __init__(self, position, n_features):
        self.position = position
        self.velocity = np.zeros_like(position)
        self.best_position = position.copy()
        self.best_score = float("-inf")  # Calinski-Harabasz Index is maximized


class PSOClustering:
    """
    A class for performing clustering using Particle Swarm Optimization (PSO) algorithm.
    """

    def __init__(self, n_clusters: int, n_particles: int, max_iter: int):
        """
        Initialize the PSOClustering with the given parameters.

        Args:
            n_clusters (int): Number of clusters.
            n_particles (int): Number of particles.
            max_iter (int): Maximum number of iterations.
        """
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(self, X: np.ndarray) -> tuple:
        """
        Optimize the cluster centroids using PSO.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            tuple: Best cluster centroids and corresponding Calinski-Harabasz Index.
        """
        n_features = X.shape[1]
        initial_centroids = dbscan_initialization(X, self.n_clusters)
        particles = [
            Particle(
                initial_centroids + np.random.randn(*initial_centroids.shape),
                n_features,
            )
            for _ in range(self.n_particles)
        ]
        global_best_position = None
        global_best_score = float("-inf")

        convergence_data = []
        diversity_data = []

        # Determine feature boundaries
        min_bound = np.min(X, axis=0)
        max_bound = np.max(X, axis=0)

        start_time = time.time()

        for iteration in range(self.max_iter):
            iteration_scores = []
            for particle in particles:
                distances = np.linalg.norm(
                    X[:, np.newaxis, :] - particle.position, axis=2
                )
                labels = np.argmin(distances, axis=1)

                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    calinski_score = calinski_harabasz_score(X, labels)

                    intra_cluster_var = np.mean(
                        [
                            np.var(X[labels == label], axis=0).sum()
                            for label in unique_labels
                        ]
                    )
                    calinski_score /= 1 + intra_cluster_var

                    deviation = abs(len(unique_labels) - self.n_clusters)
                    cluster_penalty = (1 / (1 + deviation)) ** 2
                    calinski_score *= cluster_penalty
                else:
                    calinski_score = float("-inf")

                iteration_scores.append(calinski_score)

                if calinski_score > particle.best_score:
                    particle.best_score = calinski_score
                    particle.best_position = particle.position.copy()

                if calinski_score > global_best_score:
                    global_best_score = calinski_score
                    global_best_position = particle.position.copy()

            convergence_data.append(global_best_score)

            particle_positions = np.array([particle.position for particle in particles])
            diversity = np.std(particle_positions, axis=0).mean()
            diversity_data.append(diversity)

            # Adaptive inertia weight based on diversity
            w = 0.9 - (0.5 * (iteration / self.max_iter))
            if diversity < 0.1:
                w += 0.1

            c1 = 1.5
            c2 = 2.0

            if diversity < 0.1:
                c1 += 0.5
                c2 -= 0.5
            else:
                c1 -= 0.5
                c2 += 0.5

            max_velocity = np.abs(max_bound - min_bound) * 0.1
            for particle in particles:
                r1, r2 = random.random(), random.random()
                particle.velocity = (
                    w * particle.velocity
                    + c1 * r1 * (particle.best_position - particle.position)
                    + c2 * r2 * (global_best_position - particle.position)
                )
                particle.velocity = np.clip(
                    particle.velocity, -max_velocity, max_velocity
                )
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, min_bound, max_bound)

            # Apply mutation if diversity is low
            mutation_rate = 0.1
            if diversity < 0.1:
                for particle in particles:
                    if random.random() < mutation_rate:
                        mutation_strength = np.random.randn(*particle.position.shape)
                        particle.position += mutation_strength
                        particle.position = np.clip(
                            particle.position, min_bound, max_bound
                        )

            if iteration % 100 == 0:
                self.logger.info(f"Iteration {iteration}/{self.max_iter} complete.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"PSO Clustering with Calinski-Harabasz Index took {elapsed_time:.2f} seconds."
        )

        self.logger.info("PSO optimization complete.")
        return global_best_position, global_best_score, convergence_data, diversity_data

    def run(self, X: np.ndarray) -> tuple:
        """
        Run the PSO clustering algorithm.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            tuple: Cluster labels and results.
        """
        best_centroids, best_calinski, convergence_data, diversity_data = self.optimize(
            X
        )
        distances = np.linalg.norm(X[:, np.newaxis, :] - best_centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        eval_metrics = Evaluation.evaluate(X, labels)

        self.logger.info("\nPSO Clustering Results:")
        results = {
            "Algorithm": "PSO",
            "Calinski Harabasz Index": best_calinski,
            "Silhouette Score": eval_metrics["silhouette_score"],
            "Davies Bouldin Index": eval_metrics["davies_bouldin_index"],
            "Dunn Index": eval_metrics["dunn_index"],
            "Convergence Data": convergence_data,
            "Diversity Data": diversity_data,
        }
        self.logger.info(results)
        return labels, results


class GeneticAlgorithmClustering:
    """
    A class for performing clustering using Genetic Algorithm (GA).
    """

    def __init__(
        self,
        n_clusters: int,
        n_population: int,
        max_iter: int,
        mutation_rate: float = 0.1,
    ):
        """
        Initialize the GeneticAlgorithmClustering with the given parameters.

        Args:
            n_clusters (int): Number of clusters.
            n_population (int): Population size.
            max_iter (int): Maximum number of iterations.
            mutation_rate (float): Mutation rate.
        """
        self.n_clusters = n_clusters
        self.n_population = n_population
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.logger = logging.getLogger(self.__class__.__name__)

    def initialize_population(self, X: np.ndarray) -> list:
        """
        Initialize the population for GA using DBSCAN.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            list: Initialized population.
        """
        initial_centroids = dbscan_initialization(X, self.n_clusters)
        return [
            initial_centroids + np.random.randn(*initial_centroids.shape)
            for _ in range(self.n_population)
        ]

    def mutate(self, individual: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Mutate an individual.

        Args:
            individual (np.ndarray): Individual to mutate.
            X (np.ndarray): Data to cluster.

        Returns:
            np.ndarray: Mutated individual.
        """
        mutation = np.random.randn(*individual.shape) * self.mutation_rate
        return individual + mutation

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform crossover between two parents.

        Args:
            parent1 (np.ndarray): First parent.
            parent2 (np.ndarray): Second parent.

        Returns:
            np.ndarray: Offspring.
        """
        return (parent1 + parent2) / 2

    def optimize(self, X: np.ndarray) -> tuple:
        """
        Optimize the cluster centroids using GA.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            tuple: Best cluster centroids and corresponding Calinski-Harabasz Index.
        """
        population = self.initialize_population(X)
        convergence_data = []
        diversity_data = []

        for iteration in range(self.max_iter):
            fitness_scores = []
            for individual in population:
                distances = np.linalg.norm(X[:, np.newaxis, :] - individual, axis=2)
                labels = np.argmin(distances, axis=1)

                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    calinski_score = calinski_harabasz_score(X, labels)
                    intra_cluster_var = np.mean(
                        [
                            np.var(X[labels == label], axis=0).sum()
                            for label in unique_labels
                        ]
                    )
                    calinski_score /= 1 + intra_cluster_var
                    deviation = abs(len(unique_labels) - self.n_clusters)
                    cluster_penalty = (1 / (1 + deviation)) ** 2
                    calinski_score *= cluster_penalty
                else:
                    calinski_score = float("-inf")

                fitness_scores.append(calinski_score)

            sorted_indices = np.argsort(fitness_scores)[
                ::-1
            ]  # Sort in descending order
            population = [population[i] for i in sorted_indices]

            convergence_data.append(max(fitness_scores))

            # Corrected diversity calculation
            population_array = np.array(population)
            diversity = np.mean(np.std(population_array, axis=0))
            diversity_data.append(diversity)

            next_generation = population[: self.n_population // 2]
            while len(next_generation) < self.n_population:
                parent1, parent2 = random.sample(next_generation, 2)
                child = self.crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child, X)
                next_generation.append(child)

            population = next_generation

            if iteration % 100 == 0:
                self.logger.info(f"Iteration {iteration}/{self.max_iter} complete.")

        best_individual = population[0]
        best_score = fitness_scores[0]
        self.logger.info("Genetic Algorithm optimization complete.")
        return best_individual, best_score, convergence_data, diversity_data

    def run(self, X: np.ndarray) -> tuple:
        """
        Run the GA clustering algorithm.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            tuple: Cluster labels and results.
        """
        best_centroids, best_calinski, convergence_data, diversity_data = self.optimize(
            X
        )
        distances = np.linalg.norm(X[:, np.newaxis, :] - best_centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        eval_metrics = Evaluation.evaluate(X, labels)

        self.logger.info("\nGenetic Algorithm Clustering Results:")
        results = {
            "Algorithm": "Genetic Algorithm",
            "Calinski Harabasz Index": best_calinski,
            "Silhouette Score": eval_metrics["silhouette_score"],
            "Davies Bouldin Index": eval_metrics["davies_bouldin_index"],
            "Dunn Index": eval_metrics["dunn_index"],
            "Convergence Data": convergence_data,
            "Diversity Data": diversity_data,
        }
        self.logger.info(results)
        return labels, results


class SimulatedAnnealingClustering:
    def __init__(self, n_clusters, initial_temp, final_temp, alpha, max_iter):
        self.n_clusters = n_clusters
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, X):
        def objective_function(centroids):
            centroids = centroids.reshape(self.n_clusters, -1)
            distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                calinski_score = calinski_harabasz_score(X, labels)
                intra_cluster_var = np.mean(
                    [
                        np.var(X[labels == label], axis=0).sum()
                        for label in unique_labels
                    ]
                )
                calinski_score /= 1 + intra_cluster_var
                deviation = abs(len(unique_labels) - self.n_clusters)
                cluster_penalty = (1 / (1 + deviation)) ** 2
                calinski_score *= cluster_penalty
            else:
                calinski_score = float("-inf")
            return (
                -calinski_score
            )  # Dual annealing minimizes, so return the negative of the score

        n_features = X.shape[1]
        initial_centroids = dbscan_initialization(X, self.n_clusters)
        bounds = [
            (np.min(X[:, j]), np.max(X[:, j]))
            for j in range(n_features)
            for _ in range(self.n_clusters)
        ]

        convergence_data = []
        diversity_data = []

        def callback(x, f, context):
            centroids = x.reshape(self.n_clusters, n_features)
            distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            convergence_data.append(-f)
            diversity = np.std(distances, axis=0).mean()
            diversity_data.append(diversity)

        result = dual_annealing(
            objective_function,
            bounds,
            maxiter=self.max_iter,
            x0=initial_centroids.flatten(),
            callback=callback,
        )
        best_centroids = result.x.reshape(self.n_clusters, n_features)

        distances = np.linalg.norm(X[:, np.newaxis, :] - best_centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        eval_metrics = Evaluation.evaluate(X, labels)

        self.logger.info("\nSimulated Annealing Clustering Results:")
        results = {
            "Algorithm": "Simulated Annealing",
            "Calinski Harabasz Index": -result.fun,  # Convert back to positive
            "Silhouette Score": eval_metrics["silhouette_score"],
            "Davies Bouldin Index": eval_metrics["davies_bouldin_index"],
            "Dunn Index": eval_metrics["dunn_index"],
            "Convergence Data": convergence_data,
            "Diversity Data": diversity_data,
        }
        self.logger.info(results)
        return labels, results


class TabuSearchClustering:
    """
    A class for performing clustering using Tabu Search (TS) algorithm.
    """

    def __init__(self, n_clusters: int, max_iter: int, tabu_tenure: int):
        """
        Initialize the TabuSearchClustering with the given parameters.

        Args:
            n_clusters (int): Number of clusters.
            max_iter (int): Maximum number of iterations.
            tabu_tenure (int): Tabu tenure.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.logger = logging.getLogger(self.__class__.__name__)

    def initialize_solution(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize the solution for TS using DBSCAN.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            np.ndarray: Initial solution.
        """
        return dbscan_initialization(X, self.n_clusters)

    def get_neighbors(self, solution: np.ndarray, X: np.ndarray) -> list:
        """
        Get neighbors of a solution.

        Args:
            solution (np.ndarray): Current solution.
            X (np.ndarray): Data to cluster.

        Returns:
            list: List of neighbors.
        """
        neighbors = []
        for i in range(len(solution)):
            new_solution = solution.copy()
            new_solution[i] = X[random.randint(0, len(X) - 1)]
            neighbors.append(new_solution)
        return neighbors

    def evaluate(self, solution: np.ndarray, X: np.ndarray) -> float:
        """
        Evaluate a solution.

        Args:
            solution (np.ndarray): Solution to evaluate.
            X (np.ndarray): Data to cluster.

        Returns:
            float: Evaluation score (negative Calinski-Harabasz Index).
        """
        distances = np.linalg.norm(X[:, np.newaxis, :] - solution, axis=2)
        labels = np.argmin(distances, axis=1)

        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            calinski_score = calinski_harabasz_score(X, labels)
            intra_cluster_var = np.mean(
                [np.var(X[labels == label], axis=0).sum() for label in unique_labels]
            )
            calinski_score /= 1 + intra_cluster_var
            deviation = abs(len(unique_labels) - self.n_clusters)
            cluster_penalty = (1 / (1 + deviation)) ** 2
            calinski_score *= cluster_penalty
        else:
            calinski_score = float("-inf")
        return -calinski_score

    def optimize(self, X: np.ndarray) -> tuple:
        """
        Optimize the cluster centroids using TS.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            tuple: Best cluster centroids and corresponding Calinski-Harabasz Index.
        """
        current_solution = self.initialize_solution(X)
        current_score = self.evaluate(current_solution, X)
        best_solution = current_solution
        best_score = current_score
        tabu_list = []

        convergence_data = []
        diversity_data = []

        for iteration in range(self.max_iter):
            neighbors = self.get_neighbors(current_solution, X)
            neighbors_scores = [self.evaluate(neighbor, X) for neighbor in neighbors]

            best_neighbor_index = np.argmin(neighbors_scores)
            best_neighbor = neighbors[best_neighbor_index]
            best_neighbor_score = neighbors_scores[best_neighbor_index]

            if best_neighbor_score < best_score:
                best_solution = best_neighbor
                best_score = best_neighbor_score

            tabu_list.append(current_solution.tolist())
            if len(tabu_list) > self.tabu_tenure:
                tabu_list.pop(0)

            if best_neighbor.tolist() not in tabu_list:
                current_solution = best_neighbor
                current_score = best_neighbor_score

            convergence_data.append(-best_score)

            # Corrected diversity calculation
            neighbors_array = np.array(neighbors)
            diversity = np.mean(np.std(neighbors_array, axis=0))
            diversity_data.append(diversity)

            if iteration % 100 == 0:
                self.logger.info(f"Iteration {iteration}/{self.max_iter} complete.")

        self.logger.info("Tabu Search optimization complete.")
        return best_solution, -best_score, convergence_data, diversity_data

    def run(self, X: np.ndarray) -> tuple:
        """
        Run the TS clustering algorithm.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            tuple: Cluster labels and results.
        """
        best_centroids, best_calinski, convergence_data, diversity_data = self.optimize(
            X
        )
        distances = np.linalg.norm(X[:, np.newaxis, :] - best_centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        eval_metrics = Evaluation.evaluate(X, labels)

        self.logger.info("\nTabu Search Clustering Results:")
        results = {
            "Algorithm": "Tabu Search",
            "Calinski Harabasz Index": best_calinski,
            "Silhouette Score": eval_metrics["silhouette_score"],
            "Davies Bouldin Index": eval_metrics["davies_bouldin_index"],
            "Dunn Index": eval_metrics["dunn_index"],
            "Convergence Data": convergence_data,
            "Diversity Data": diversity_data,
        }
        self.logger.info(results)
        return labels, results
