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

# To run:
#   - Python 2: `PYSPARK_PYTHON=python2 spark-submit --master local[*] --driver-class-path SystemML.jar test_matrix_agg_fn.py`
#   - Python 3: `PYSPARK_PYTHON=python3 spark-submit --master local[*] --driver-class-path SystemML.jar test_matrix_agg_fn.py`

# Make the `systemml` package importable
import os
import sys
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)

import unittest
import systemml as sml
import numpy as np
from scipy.stats import kurtosis, skew, moment
from pyspark.context import SparkContext
sc = SparkContext()

dim = 5
m1 = np.array(np.random.randint(100, size=dim*dim) + 1.01, dtype=np.double)
m1.shape = (dim, dim)
m2 = np.array(np.random.randint(5, size=dim*dim) + 1, dtype=np.double)
m2.shape = (dim, dim)
s = 3.02

class TestMatrixAggFn(unittest.TestCase):

    def test_sum1(self):
        self.assertTrue(np.allclose(sml.matrix(m1).sum(), m1.sum()))

    def test_sum2(self):
        self.assertTrue(np.allclose(sml.matrix(m1).sum(axis=0), m1.sum(axis=0)))
    
    def test_sum3(self):
        self.assertTrue(np.allclose(sml.matrix(m1).sum(axis=1), m1.sum(axis=1).reshape(dim, 1)))

    def test_mean1(self):
        self.assertTrue(np.allclose(sml.matrix(m1).mean(), m1.mean()))

    def test_mean2(self):
        self.assertTrue(np.allclose(sml.matrix(m1).mean(axis=0), m1.mean(axis=0).reshape(1, dim)))
    
    def test_mean3(self):
        self.assertTrue(np.allclose(sml.matrix(m1).mean(axis=1), m1.mean(axis=1).reshape(dim, 1)))
    
    def test_hstack(self):
        self.assertTrue(np.allclose(sml.matrix(m1).hstack(sml.matrix(m1)), np.hstack((m1, m1))))    
    
    def test_vstack(self):
        self.assertTrue(np.allclose(sml.matrix(m1).vstack(sml.matrix(m1)), np.vstack((m1, m1))))
        
    def test_full(self):
        self.assertTrue(np.allclose(sml.full((2, 3), 10.1), np.full((2, 3), 10.1)))
    
    def test_seq(self):
        self.assertTrue(np.allclose(sml.seq(3), np.arange(3+1).reshape(4, 1)))
        
    def test_var1(self):
        print(str(np.array(sml.matrix(m1).var())) + " " + str(np.array(m1.var(ddof=1))))
        self.assertTrue(np.allclose(sml.matrix(m1).var(), m1.var(ddof=1)))

    def test_var2(self):
        self.assertTrue(np.allclose(sml.matrix(m1).var(axis=0), m1.var(axis=0, ddof=1).reshape(1, dim)))
    
    def test_var3(self):
        self.assertTrue(np.allclose(sml.matrix(m1).var(axis=1), m1.var(axis=1, ddof=1).reshape(dim, 1)))
    
    def test_moment3(self):
        self.assertTrue(np.allclose(sml.matrix(m1).moment(moment=3, axis=None), moment(m1, moment=3, axis=None)))
        
    def test_moment4(self):
        self.assertTrue(np.allclose(sml.matrix(m1).moment(moment=4, axis=None), moment(m1, moment=4, axis=None)))

if __name__ == "__main__":
    unittest.main()
