import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile
import requests
import io
import logging


class Dataset:
    """
    Base class for datasets.
    """

    def __init__(self, numerical_features: list):
        """
        Initialize the dataset with numerical features.

        Args:
            numerical_features (list): List of numerical feature names.
        """
        self.numerical_features = numerical_features
        self.data = None
        self.X_scaled = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess(self):
        """
        Preprocess the dataset. Must be implemented by subclasses.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError(
            "Each dataset must implement its own preprocess method."
        )

    def standardize(self, X: pd.DataFrame) -> np.ndarray:
        """
        Standardize the numerical features of the dataset.

        Args:
            X (pd.DataFrame): Dataframe containing numerical features.

        Returns:
            np.ndarray: Standardized numerical features.
        """
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)
        self.logger.info("Data standardized.")
        return self.X_scaled


class StudentDataset(Dataset):
    """
    Student performance dataset.
    """

    def __init__(self):
        """
        Initialize the StudentDataset with the URL and file name.
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
        file_name = "student-mat.csv"
        # numerical_features = ["age", "absences", "G1", "G2", "G3"]
        numerical_features = [
            "age",
            "Medu",
            "Fedu",
            "traveltime",
            "studytime",
            "failures",
            "famrel",
            "freetime",
            "goout",
            "Dalc",
            "Walc",
            "health",
            "absences",
            "G1",
            "G2",
            "G3",
        ]
        super().__init__(numerical_features)
        self.url = url
        self.file_name = file_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self):
        """
        Load the dataset from the URL.
        """
        self.logger.info(f"Loading data from {self.url}")
        response = requests.get(self.url)
        with ZipFile(io.BytesIO(response.content)) as z:
            with z.open(self.file_name) as f:
                self.data = pd.read_csv(f, sep=";")
        self.logger.info("Data loaded successfully.")

    def preprocess(self) -> np.ndarray:
        """
        Preprocess the student dataset by loading and standardizing the data.

        Returns:
            np.ndarray: Standardized numerical features.
        """
        self.load_data()

        # Define the numerical features to include
        numerical_features = ["age", "absences", "G2", "G3"]

        # Select the specified numerical features
        X = self.data[numerical_features]
        self.logger.info("Preprocessing data.")
        return self.standardize(X)


class WineDataset(Dataset):
    """
    Wine dataset.
    """

    def __init__(self):
        """
        Initialize the WineDataset with the URL and numerical features.
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
        numerical_features = [
            "Alcohol",
            "Malic acid",
            "Ash",
            "Alcalinity of ash",
            "Magnesium",
            "Total phenols",
            "Flavanoids",
            "Nonflavanoid phenols",
            "Proanthocyanins",
            "Color intensity",
            "Hue",
            "OD280/OD315 of diluted wines",
            "Proline",
        ]
        super().__init__(numerical_features)
        self.url = url
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self):
        """
        Load the dataset from the URL.
        """
        self.logger.info(f"Loading data from {self.url}")
        self.data = pd.read_csv(self.url, header=None)
        self.data.columns = ["Class"] + self.numerical_features
        self.logger.info("Data loaded successfully.")

    def preprocess(self) -> np.ndarray:
        """
        Preprocess the wine dataset by loading and standardizing the data.

        Returns:
            np.ndarray: Standardized numerical features.
        """
        self.load_data()

        # Define the numerical features to include
        numerical_features = [
            "Flavanoids",
            "Proline",
            "OD280/OD315 of diluted wines",
            "Color intensity",
            "Alcohol",
            "Hue",
        ]

        # Select the specified numerical features
        X = self.data[numerical_features]
        self.logger.info("Preprocessing data.")
        return self.standardize(X)


class IrisDataset(Dataset):
    """
    Iris flower dataset.
    """

    def __init__(self):
        """
        Initialize the IrisDataset with the URL and numerical features.
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        numerical_features = [
            "sepal length",
            "sepal width",
            "petal length",
            "petal width",
        ]
        super().__init__(numerical_features)
        self.url = url
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self):
        """
        Load the dataset from the URL.
        """
        self.logger.info(f"Loading data from {self.url}")
        self.data = pd.read_csv(self.url, header=None)
        self.data.columns = self.numerical_features + ["Class"]
        self.logger.info("Data loaded successfully.")

    def preprocess(self) -> np.ndarray:
        """
        Preprocess the iris dataset by loading and standardizing the data.

        Returns:
            np.ndarray: Standardized numerical features.
        """
        self.load_data()
        X = self.data[self.numerical_features]
        self.logger.info("Preprocessing data.")
        return self.standardize(X)


class BreastCancerDataset(Dataset):
    """
    Breast cancer Wisconsin dataset.
    """

    def __init__(self):
        """
        Initialize the BreastCancerDataset with the URL and numerical features.
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        numerical_features = [
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
        ]

        super().__init__(numerical_features)
        self.url = url
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self):
        """
        Load the dataset from the URL.
        """
        self.logger.info(f"Loading data from {self.url}")
        self.data = pd.read_csv(self.url, header=None)
        self.data.columns = ["ID", "Diagnosis"] + self.numerical_features
        self.data.drop(columns=["ID"], inplace=True)
        self.logger.info("Data loaded successfully.")

    def preprocess(self) -> np.ndarray:
        """
        Preprocess the breast cancer dataset by loading and standardizing the data.

        Returns:
            np.ndarray: Standardized numerical features.
        """
        self.load_data()

        # Define the numerical features to include
        numerical_features = [
            "radius_mean",
            "concavity_worst",
            "compactness_worst",
            "texture_mean",
            "radius_se",
            "texture_se",
        ]

        # Select the specified numerical features
        X = self.data[numerical_features]
        self.logger.info("Preprocessing data.")
        return self.standardize(X)
