# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

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
            'TRUE' if self.params.get('verbose', False) else 'FALSE',
            50,  # avg_sample_size_per_centroid unkown in sklearn
            -1 if self.params['random_state'] is None \
            else self.params['random_state']
        ]


class DBSCANMapper(Mapper):
    name = 'dbscan'
    sklearn_name = 'dbscan'
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
    sklearn_name = 'gaussianmixture'
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
            f'"{self.model_map.get(self.params.get("covariance_type", "VVV"))}"',
            f'"{self.params.get("init_params", "kmeans")}"',
            self.params.get('max_iter', 100),
            self.params.get('reg_covar', 1e-6),
            self.params.get('tol', 0.000001),
            self.params.get('seed', -1),
            self.params.get('verbose', 'FALSE')
        ]
