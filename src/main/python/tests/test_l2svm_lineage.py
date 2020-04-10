#-------------------------------------------------------------
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
#-------------------------------------------------------------

import warnings
import unittest
import re
import os
import sys
from typing import Tuple
import numpy as np

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)
from systemds.matrix import Matrix
from systemds.context import SystemDSContext

sds = SystemDSContext()


class TestAPI(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore",
                                message="unclosed",
                                category=ResourceWarning)

    def tearDown(self):
        warnings.filterwarnings(action="ignore",
                                message="unclosed",
                                category=ResourceWarning)

    def test_getl2svm_lineage(self):
        features, labels = generate_matrices_for_l2svm(10, seed=1304)
        # get the lineage trace
        lt = features.l2svm(labels).get_lineage_trace()
        with open(os.path.join("tests", "lt_l2svm.txt"), "r") as file:
            data = file.read()
        file.close()
        self.assertEqual(reVars(lt), reVars(data))

    def test_getl2svm_lineage2(self):
        features, labels = generate_matrices_for_l2svm(10, seed=1304)
        # get the lineage trace
        model, lt = features.l2svm(labels).compute(lineage=True)
        with open(os.path.join("tests", "lt_l2svm.txt"), "r") as file:
            data = file.read()
        file.close()
        self.assertEqual(reVars(lt), reVars(data))


def generate_matrices_for_l2svm(dims: int, seed: int = 1234) -> Tuple[Matrix, Matrix]:
    np.random.seed(seed)
    m1 = np.array(np.random.randint(100, size=dims * dims) + 1.01, dtype=np.double)
    m1.shape = (dims, dims)
    m2 = np.zeros((dims, 1))
    for i in range(dims):
        if np.random.random() > 0.5:
            m2[i][0] = 1
    return sds.matrix(m1), sds.matrix(m2)


def reVars(s: str) -> str:
    s = re.sub(r'\b_mVar\d*\b', '', s)
    s = re.sub(r'\b_Var\d*\b', '', s)
    return s


if __name__ == "__main__":
    unittest.main(exit=False)
    sds.close()
