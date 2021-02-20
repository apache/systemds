from .mapper import Mapper


class KmeansMapper(Mapper):
    name = 'kmeans'
    sklearn_name = 'kmeans'
    mapped_output = [
        'C',  # The output matrix with the centroids
        'Y'  # The mapping of records to centroids
    ]

    def map_params(self):
        self.mapped_params = [
            self.params['n_clusters'],
            self.params['n_init'],
            self.params['max_iter'],
            self.params['tol'],
            self.params.get('verbose', False),
            50,  # avg_sample_size_per_centroid unkown in sklearn
            -1 if self.params['random_state'] is None \
            else self.params['random_state']
        ]


class DBSCANMapper(Mapper):
    name = 'dbscan'
    mapped_output = [
        'clusterMembers'
    ]

    def map_params(self):
        self.mapped_params = [
            self.params.get('eps', 0.5),
            self.params.get('min_samples', 5)
        ]


class GaussianMixtureMapper(Mapper):
    name = 'gmm'
    model_map = {
        'full': 'VVV',
        'tied': 'EEE',
        'diag': 'VVI',
        'spherical': 'VVI'
    }
    mapped_output = [
        'weight',
        'labels',
        'df',
        'bic'
    ]

    def map_params(self):
        self.mapped_params = [
            self.params.get('n_components', 3),
            self.model_map.get(self.params.get('covariance_type', 'VVV')),
            self.params.get('init_params', 'kmeans'),
            self.params.get('max_iter', 100),
            self.params.get('reg_covar', 1e-6),
            self.params.get('tol', 0.000001)
        ]
