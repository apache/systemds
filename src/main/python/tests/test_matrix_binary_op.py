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
#   - Python 2: `PYSPARK_PYTHON=python2 spark-submit --master local[*] --driver-class-path SystemML.jar test_matrix_binary_op.py`
#   - Python 3: `PYSPARK_PYTHON=python3 spark-submit --master local[*] --driver-class-path SystemML.jar test_matrix_binary_op.py`

# Make the `systemml` package importable
import os
import sys
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)

import unittest
import systemml as sml
import numpy as np
from pyspark.context import SparkContext
sc = SparkContext()

dim = 5
m1 = np.array(np.random.randint(100, size=dim*dim) + 1.01, dtype=np.double)
m1.shape = (dim, dim)
m2 = np.array(np.random.randint(5, size=dim*dim) + 1, dtype=np.double)
m2.shape = (dim, dim)
s = 3.02

class TestBinaryOp(unittest.TestCase):

    def test_plus(self):
        self.assertTrue(np.allclose(sml.matrix(m1) + sml.matrix(m2), m1 + m2))
        
    def test_minus(self):
        self.assertTrue(np.allclose(sml.matrix(m1) - sml.matrix(m2), m1 - m2))
        
    def test_mul(self):
        self.assertTrue(np.allclose(sml.matrix(m1) * sml.matrix(m2), m1 * m2))
    
    def test_div(self):
        self.assertTrue(np.allclose(sml.matrix(m1) / sml.matrix(m2), m1 / m2))
    
    #def test_power(self):
    #    self.assertTrue(np.allclose(sml.matrix(m1) ** sml.matrix(m2), m1 ** m2))
    
    def test_plus1(self):
        self.assertTrue(np.allclose(sml.matrix(m1) + m2, m1 + m2))
        
    def test_minus1(self):
        self.assertTrue(np.allclose(sml.matrix(m1) - m2, m1 - m2))
        
    def test_mul1(self):
        self.assertTrue(np.allclose(sml.matrix(m1) * m2, m1 * m2))
    
    def test_div1(self):
        self.assertTrue(np.allclose(sml.matrix(m1) / m2, m1 / m2))
    
    def test_power1(self):
        self.assertTrue(np.allclose(sml.matrix(m1) ** m2, m1 ** m2))
        
    def test_plus2(self):
        self.assertTrue(np.allclose(m1 + sml.matrix(m2), m1 + m2))
        
    def test_minus2(self):
        self.assertTrue(np.allclose(m1 - sml.matrix(m2), m1 - m2))
        
    def test_mul2(self):
        self.assertTrue(np.allclose(m1 * sml.matrix(m2), m1 * m2))
    
    def test_div2(self):
        self.assertTrue(np.allclose(m1 / sml.matrix(m2), m1 / m2))
    
    def test_power2(self):
        self.assertTrue(np.allclose(m1 ** sml.matrix(m2), m1 ** m2))
    
    def test_plus3(self):
        self.assertTrue(np.allclose(sml.matrix(m1) + s, m1 + s))
        
    def test_minus3(self):
        self.assertTrue(np.allclose(sml.matrix(m1) - s, m1 - s))
        
    def test_mul3(self):
        self.assertTrue(np.allclose(sml.matrix(m1) * s, m1 * s))
    
    def test_div3(self):
        self.assertTrue(np.allclose(sml.matrix(m1) / s, m1 / s))
    
    def test_power3(self):
        self.assertTrue(np.allclose(sml.matrix(m1) ** s, m1 ** s))
    
    def test_plus4(self):
        self.assertTrue(np.allclose(s + sml.matrix(m2), s + m2))
        
    def test_minus4(self):
        self.assertTrue(np.allclose(s - sml.matrix(m2), s - m2))
        
    def test_mul4(self):
        self.assertTrue(np.allclose(s * sml.matrix(m2), s * m2))
    
    def test_div4(self):
        self.assertTrue(np.allclose(s / sml.matrix(m2), s / m2))
    
    def test_power4(self):
        self.assertTrue(np.allclose(s ** sml.matrix(m2), s ** m2))

    def test_lt(self):
        self.assertTrue(np.allclose(sml.matrix(m1) < sml.matrix(m2), m1 < m2))
        
    def test_gt(self):
        self.assertTrue(np.allclose(sml.matrix(m1) > sml.matrix(m2), m1 > m2))
        
    def test_le(self):
        self.assertTrue(np.allclose(sml.matrix(m1) <= sml.matrix(m2), m1 <= m2))
    
    def test_ge(self):
        self.assertTrue(np.allclose(sml.matrix(m1) >= sml.matrix(m2), m1 >= m2))
        
    def test_abs(self):
        self.assertTrue(np.allclose(sml.matrix(m1).abs(), np.abs(m1)))

if __name__ == "__main__":
    unittest.main()
