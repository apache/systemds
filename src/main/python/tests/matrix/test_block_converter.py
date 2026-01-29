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


import unittest

import numpy as np
from py4j.java_gateway import JVMView
from systemds.context import SystemDSContext
from systemds.utils.converters import matrix_block_to_numpy, numpy_to_matrix_block
import scipy.sparse as sp


class Test_MatrixBlockConverter(unittest.TestCase):
    """Test class for testing behavior of the fundamental DMLScript class"""

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(
            capture_stdout=True, logging_level=50, data_transfer_mode=0
        )

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_unrelated_java_code(self):
        # https://www.py4j.org/getting_started.html
        java_list = self.sds.java_gateway.jvm.java.util.ArrayList()
        java_list.append(10)
        java_list.append(131)
        java_list.append(31)
        self.sds.java_gateway.jvm.java.util.Collections.sort(java_list)
        self.assertListEqual([10, 31, 131], eval(java_list.toString()))

    def test_simple_1x1(self):
        self.convert_back_and_forth(array=np.array([1]))

    def test_simple_1x1_redundant_dimension(self):
        self.convert_back_and_forth(np.array([[1]]))

    def test_random_nxn(self):
        n = 10
        rng = np.random.default_rng(seed=7)
        array = rng.standard_normal(n)
        array = array * np.array([array]).T
        self.convert_back_and_forth(array)

    def test_random_nxk(self):
        n = 10
        k = 3
        rng = np.random.default_rng(seed=7)
        array = np.array([rng.standard_normal(n) for x in range(k)])
        self.convert_back_and_forth(array)

    def test_random_sparse_csr_nxn(self):
        n = 10
        array = sp.rand(n, n, density=0.1, format="csr")
        self.convert_back_and_forth(array)

    def test_sparse_csr_rectangular(self):
        """Test CSR conversion with rectangular matrices"""
        array = sp.rand(5, 10, density=0.2, format="csr")
        self.convert_back_and_forth(array)

    def test_sparse_csr_known_values(self):
        """Test CSR conversion with a known sparse matrix"""
        # Create a known CSR matrix
        data = np.array([1.0, 2.0, 3.0, 4.0])
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 2, 1, 2])
        array = sp.csr_matrix((data, (row, col)), shape=(3, 3))
        self.convert_back_and_forth(array)

    def test_empty_dense_0x0(self):
        """Test conversion of empty 0x0 dense matrix"""
        array = np.array([]).reshape(0, 0)
        self.convert_back_and_forth(array)

    def test_empty_dense_0xn(self):
        """Test conversion of empty matrix with zero rows"""
        array = np.array([]).reshape(0, 5)
        self.convert_back_and_forth(array)

    def test_empty_dense_nx0(self):
        """Test conversion of empty matrix with zero columns"""
        array = np.array([]).reshape(5, 0)
        self.convert_back_and_forth(array)

    def test_sparse_csr_empty_rows(self):
        """Test CSR conversion with rows that have no non-zero entries"""
        # 4x3 matrix: row 0 has 2 entries, row 1 is empty, row 2 has 1 entry, row 3 is empty
        data = np.array([1.0, 2.0, 3.0])
        row = np.array([0, 0, 2])
        col = np.array([0, 2, 1])
        array = sp.csr_matrix((data, (row, col)), shape=(4, 3))
        self.convert_back_and_forth(array)

    def test_sparse_csr_first_and_last_row_empty(self):
        """Test CSR with first and last rows empty"""
        data = np.array([1.0, 2.0])
        row = np.array([1, 1])
        col = np.array([0, 1])
        array = sp.csr_matrix((data, (row, col)), shape=(3, 2))
        self.convert_back_and_forth(array)

    def test_sparse_csr_all_zeros(self):
        """Test CSR with no non-zero entries (empty structure)"""
        array = sp.csr_matrix((3, 4))
        self.convert_back_and_forth(array)

    def test_sparse_coo_empty_rows(self):
        """Test COO conversion with empty rows"""
        data = np.array([1.0, 2.0])
        row = np.array([0, 2])
        col = np.array([1, 1])
        array = sp.coo_matrix((data, (row, col)), shape=(4, 2))
        self.convert_back_and_forth(array)

    def test_dense_all_zeros(self):
        """Test dense matrix with all zeros"""
        array = np.zeros((4, 3))
        self.convert_back_and_forth(array)

    def test_dense_single_row(self):
        """Test dense matrix with single row (1xn)"""
        array = np.array([[1.0, 2.0, 3.0]])
        self.convert_back_and_forth(array)

    def test_dense_single_column(self):
        """Test dense matrix with single column (nx1)"""
        array = np.array([[1.0], [2.0], [3.0]])
        self.convert_back_and_forth(array)

    def test_random_sparse_coo_nxn(self):
        n = 10
        array = sp.rand(n, n, density=0.1, format="coo")
        self.convert_back_and_forth(array)

    def test_sparse_csr_empty_0x0(self):
        """Test empty 0x0 CSR matrix"""
        array = sp.csr_matrix((0, 0))
        self.convert_back_and_forth(array)

    def test_sparse_csr_empty_0xn(self):
        """Test CSR with zero rows"""
        array = sp.csr_matrix((0, 4))
        self.convert_back_and_forth(array)

    def test_sparse_csr_empty_nx0(self):
        """Test CSR with zero columns"""
        array = sp.csr_matrix((4, 0))
        self.convert_back_and_forth(array)

    def test_sparse_csr_single_element(self):
        """Test 1x1 CSR with one non-zero"""
        array = sp.csr_matrix(([3.14], ([0], [0])), shape=(1, 1))
        self.convert_back_and_forth(array)

    def test_sparse_csr_single_row(self):
        """Test 1xn CSR (single row)"""
        array = sp.csr_matrix(([1.0, 2.0], ([0, 0], [0, 2])), shape=(1, 4))
        self.convert_back_and_forth(array)

    def test_sparse_csr_single_column(self):
        """Test nx1 CSR (single column)"""
        array = sp.csr_matrix(([1.0, 2.0], ([0, 2], [0, 0])), shape=(4, 1))
        self.convert_back_and_forth(array)

    def test_sparse_csr_empty_columns(self):
        """Test CSR where some columns have no non-zero entries"""
        # 3x4 matrix: only columns 0 and 2 have entries
        data = np.array([1.0, 2.0, 3.0])
        row = np.array([0, 1, 2])
        col = np.array([0, 2, 2])
        array = sp.csr_matrix((data, (row, col)), shape=(3, 4))
        self.convert_back_and_forth(array)

    def test_sparse_coo_single_element(self):
        """Test COO with single non-zero"""
        array = sp.coo_matrix(([1.0], ([2], [3])), shape=(4, 5))
        self.convert_back_and_forth(array)

    def test_sparse_coo_all_zeros(self):
        """Test COO with no non-zero entries"""
        array = sp.coo_matrix((2, 3))
        self.convert_back_and_forth(array)

    def test_sparse_csc_rectangular(self):
        """Test CSC conversion (fallback path: converted to dense in converter)"""
        array = sp.csc_matrix(([1.0, 2.0, 3.0], ([0, 1, 2], [0, 0, 1])), shape=(3, 2))
        self.convert_back_and_forth(array)

    def convert_back_and_forth(self, array):
        matrix_block = numpy_to_matrix_block(self.sds, array)
        # use the ability to call functions on matrix_block.
        returned = matrix_block_to_numpy(self.sds, matrix_block)
        if isinstance(array, sp.spmatrix):
            array = array.toarray()
        self.assertTrue(np.allclose(array, returned))


if __name__ == "__main__":
    unittest.main(exit=False)
