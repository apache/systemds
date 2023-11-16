import csv
import numpy as np

from abc import ABC, abstractmethod


class Measure(ABC):
    def __init__(self, name: str, short_name=''):
        self.file_writer = None
        self.file = None
        self.name = name
        self.api_name = name.lower().replace(' ', '_')
        self.result = None
        self.uv_distance_fct = None
        self.short_name = short_name

    def add_file(self, folder_name):
        if self.file is not None:
            self.file.close()

        self.file = open(folder_name + self.api_name + '.csv', 'w')
        self.file_writer = csv.writer(self.file, delimiter=',', lineterminator='\n')

    def __del__(self):
        if self.file is not None:
            self.file.close()

    @abstractmethod
    def compute_mv_distances(self, x, y):
        pass


class PearsonCorrelation(Measure):
    def __init__(self):
        super().__init__('Pearson Correlation', 'PC')
        self.uv_distance_fct = lambda x, y: (np.dot(x - x.mean(), y - y.mean())) / (np.std(x) * np.std(y)) / len(x)

    def compute_mv_distances(self, x, y):
        mu_x = x - x.mean(1)[:, None]
        mu_y = y - y.mean(1)[:, None]

        pc_matrix = (mu_x @ mu_y.T) / np.sqrt((mu_x ** 2).sum(1)[:, None] @ (mu_y ** 2).sum(1)[None])

        nan_mask = np.equal(pc_matrix, np.nan)
        pc_matrix[nan_mask] = 0.0

        return pc_matrix


class EuclideanDistance(Measure):
    def __init__(self):
        super().__init__('Euclidean Distance', 'ED')
        self.uv_distance_fct = lambda x, y: (np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y)))

    def compute_mv_distances(self, x, y):
        d_squared = (x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None] - 2 * (x @ y.T)
        nan_mask = np.equal(d_squared, np.nan)
        d_squared[nan_mask] = 0.0
        ed = np.sqrt(d_squared)

        return ed


class CosineSimilarity(Measure):
    def __init__(self):
        super().__init__('Cosine Similarity', 'CS')
        self.uv_distance_fct = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def compute_mv_distances(self, x, y):
        return (x @ y.T) / np.sqrt((x ** 2).sum(1)[:, None] @ ((y ** 2).sum(1)[None]))

