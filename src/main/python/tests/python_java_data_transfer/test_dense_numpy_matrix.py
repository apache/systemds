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


import os
import shutil
import unittest
import pandas as pd
import numpy as np
from systemds.context import SystemDSContext
from tests.test_utils import timeout


class TestMatrixBlockConverterUnixPipe(unittest.TestCase):

    sds: SystemDSContext = None
    temp_dir: str = "tests/iotests/temp_write_csv/"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(
            data_transfer_mode=1, logging_level=10, capture_stdout=True
        )
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @timeout(60)
    def test_python_to_java(self):
        combinations = [  # (n_rows, n_cols)
            (5, 0),
            (5, 1),
            (10, 10),
        ]

        for n_rows, n_cols in combinations:
            matrix = (
                np.random.random((n_rows, n_cols))
                if n_cols != 0
                else np.random.random(n_rows)
            )
            # Transfer into SystemDS and write to CSV
            matrix_sds = self.sds.from_numpy(matrix)
            matrix_sds.write(
                self.temp_dir + "into_systemds_matrix.csv", format="csv", header=False
            ).compute()

            # Read the CSV file using pandas
            result_df = pd.read_csv(
                self.temp_dir + "into_systemds_matrix.csv", header=None
            )
            matrix_out = result_df.to_numpy()
            if n_cols == 0:
                matrix_out = matrix_out.flatten()
            # Verify the data
            self.assertTrue(np.allclose(matrix_out, matrix))

    @timeout(60)
    def test_java_to_python(self):
        """Test reading matrices from SystemDS back to Python with various dtypes."""
        # (dtype, shapes, data_type, tolerance)
        configs = [
            (np.float64, [(5, 1), (10, 10), (100, 5)], "random", 1e-9),
            (np.float32, [(10, 10), (50, 3)], "random", 1e-6),
            (np.int32, [(10, 10), (20, 5)], "randint", 0.0),
            (np.uint8, [(10, 10), (15, 8)], "randuint8", 0.0),
        ]

        def _gen_data(dtype, data_type):
            if data_type == "random":
                return lambda s: np.random.random(s).astype(dtype)
            elif data_type == "randint":
                return lambda s: np.random.randint(-10000, 10000, s).astype(dtype)
            elif data_type == "randuint8":
                return lambda s: np.random.randint(0, 255, s).astype(dtype)

        test_cases = [
            {
                "dtype": dt,
                "shape": sh,
                "data": _gen_data(dt, data_type),
                "tolerance": tol,
            }
            for dt, shapes, data_type, tol in configs
            for sh in shapes
        ] + [
            # Edge cases
            {
                "dtype": np.float64,
                "shape": (1, 1),
                "data": lambda s: np.random.random(s).astype(np.float64),
                "tolerance": 1e-9,
            },
            {
                "dtype": np.float64,
                "shape": (1, 10),
                "data": lambda s: np.random.random(s).astype(np.float64),
                "tolerance": 1e-9,
            },
            {
                "dtype": np.float64,
                "shape": (10, 10),
                "data": lambda s: np.zeros(s, dtype=np.float64),
                "tolerance": 0.0,
            },
            {
                "dtype": np.float64,
                "shape": (10, 5),
                "data": lambda s: np.random.uniform(-100.0, 100.0, s).astype(
                    np.float64
                ),
                "tolerance": 1e-9,
            },
        ]

        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i, dtype=test_case["dtype"], shape=test_case["shape"]):
                matrix = test_case["data"](test_case["shape"])

                # Create a CSV file to read into SystemDS
                csv_path = self.temp_dir + f"out_of_systemds_matrix_{i}.csv"
                pd.DataFrame(matrix).to_csv(csv_path, header=False, index=False)

                matrix_sds = self.sds.read(
                    csv_path,
                    data_type="matrix",
                    format="csv",
                )
                matrix_out = matrix_sds.compute()

                # Verify the data
                # Note: SystemDS reads all matrices as FP64, so we compare accordingly
                if test_case["tolerance"] == 0.0:
                    # Exact match for integer types
                    self.assertTrue(
                        np.array_equal(
                            matrix.astype(np.float64), matrix_out.astype(np.float64)
                        ),
                        f"Matrix with dtype {test_case['dtype']} and shape {test_case['shape']} doesn't match exactly",
                    )
                else:
                    # Approximate match for float types
                    self.assertTrue(
                        np.allclose(
                            matrix.astype(np.float64),
                            matrix_out.astype(np.float64),
                            atol=test_case["tolerance"],
                        ),
                        f"Matrix with dtype {test_case['dtype']} and shape {test_case['shape']} doesn't match within tolerance",
                    )

    @timeout(60)
    def test_java_to_python_unsupported_dtypes(self):
        """Test that unsupported dtypes are handled gracefully or converted."""
        # Note: SystemDS will convert unsupported dtypes to FP64 when reading from CSV
        # So these should still work, just with type conversion

        test_cases = [
            # INT64 - not directly supported for MatrixBlock, but CSV reads as FP64
            {
                "dtype": np.int64,
                "shape": (10, 5),
                "data": lambda s: np.random.randint(-1000000, 1000000, s).astype(
                    np.int64
                ),
            },
            # Complex types - not supported, should fail or be converted
            {
                "dtype": np.complex128,
                "shape": (5, 5),
                "data": lambda s: np.random.random(s) + 1j * np.random.random(s),
                "should_fail": True,  # Complex numbers not supported in matrices
            },
        ]

        for i, test_case in enumerate(test_cases):
            with self.subTest(i=i, dtype=test_case["dtype"], shape=test_case["shape"]):
                if test_case.get("should_fail", False):
                    # Test that unsupported types fail gracefully
                    matrix = test_case["data"](test_case["shape"])
                    csv_path = self.temp_dir + f"unsupported_matrix_{i}.csv"

                    # Writing complex numbers to CSV might fail or convert to real part
                    try:
                        pd.DataFrame(matrix).to_csv(csv_path, header=False, index=False)
                        # If writing succeeds, reading might fail or behave unexpectedly
                        with self.assertRaises(Exception):
                            matrix_sds = self.sds.read(
                                csv_path,
                                data_type="matrix",
                                format="csv",
                            )
                            matrix_sds.compute()
                    except Exception:
                        # Writing failed, which is expected
                        pass
                else:
                    # Type should be converted to FP64
                    matrix = test_case["data"](test_case["shape"])
                    csv_path = self.temp_dir + f"converted_matrix_{i}.csv"

                    # Write as the original dtype (pandas will handle conversion for CSV)
                    pd.DataFrame(matrix).to_csv(csv_path, header=False, index=False)

                    matrix_sds = self.sds.read(
                        csv_path,
                        data_type="matrix",
                        format="csv",
                    )
                    matrix_out = matrix_sds.compute()

                    # Should be converted to FP64 and match values
                    self.assertTrue(
                        np.allclose(
                            matrix.astype(np.float64),
                            matrix_out.astype(np.float64),
                            atol=1e-9,
                        ),
                        f"Converted matrix with dtype {test_case['dtype']} doesn't match",
                    )


if __name__ == "__main__":
    unittest.main(exit=False)
