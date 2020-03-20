# ------------------------------------------------------------------------------
#  Copyright 2020 Graz University of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ------------------------------------------------------------------------------

import unittest

import os
import sys
from typing import Tuple

import numpy as np

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)
from systemds.matrix import Matrix


class TestAPI(unittest.TestCase):

    def test_10x10(self):
        features, labels = generate_matrices_for_l2svm(10, seed=1304)
        # TODO calculate reference
        model = features.l2svm(labels).compute()
        self.assertTrue(np.allclose(model, np.array([[-0.03277166],
                                                     [-0.00820981],
                                                     [0.00657115],
                                                     [0.03228764],
                                                     [-0.01685067],
                                                     [0.00892918],
                                                     [0.00945636],
                                                     [0.01514383],
                                                     [0.0713272],
                                                     [-0.05113976]])))


def generate_matrices_for_l2svm(dims: int, seed: int = 1234) -> Tuple[Matrix, Matrix]:
    np.random.seed(seed)
    m1 = np.array(np.random.randint(100, size=dims * dims) + 1.01, dtype=np.double)
    m1.shape = (dims, dims)
    m2 = np.zeros((dims, 1))
    for i in range(dims):
        if np.random.random() > 0.5:
            m2[i][0] = 1
    return Matrix(m1), Matrix(m2)


if __name__ == "__main__":
    unittest.main()
