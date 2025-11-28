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
import threading
import functools
import pandas as pd
import numpy as np
from systemds.context import SystemDSContext


def timeout(seconds):
    """Decorator to add timeout to test methods."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(
                    f"Test {func.__name__} exceeded timeout of {seconds} seconds"
                )

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


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
        test_cases = [
            # Supported dtypes - FP64 (default)
            {
                "dtype": np.float64,
                "shape": (5, 1),
                "data": lambda s: np.random.random(s).astype(np.float64),
                "tolerance": 1e-9,
            },
            {
                "dtype": np.float64,
                "shape": (10, 10),
                "data": lambda s: np.random.random(s).astype(np.float64),
                "tolerance": 1e-9,
            },
            {
                "dtype": np.float64,
                "shape": (100, 5),
                "data": lambda s: np.random.random(s).astype(np.float64),
                "tolerance": 1e-9,
            },
            # Supported dtypes - FP32 (should be converted to FP64 when read back)
            {
                "dtype": np.float32,
                "shape": (10, 10),
                "data": lambda s: np.random.random(s).astype(np.float32),
                "tolerance": 1e-6,  # Lower tolerance due to float32 precision
            },
            {
                "dtype": np.float32,
                "shape": (50, 3),
                "data": lambda s: np.random.random(s).astype(np.float32),
                "tolerance": 1e-6,
            },
            # Supported dtypes - INT32 (should be converted to FP64 when read back)
            {
                "dtype": np.int32,
                "shape": (10, 10),
                "data": lambda s: np.random.randint(-1000, 1000, s).astype(np.int32),
                "tolerance": 0.0,  # Exact match expected for integers
            },
            {
                "dtype": np.int32,
                "shape": (20, 5),
                "data": lambda s: np.random.randint(-10000, 10000, s).astype(np.int32),
                "tolerance": 0.0,
            },
            # Supported dtypes - UINT8 (should be converted to FP64 when read back)
            {
                "dtype": np.uint8,
                "shape": (10, 10),
                "data": lambda s: np.random.randint(0, 255, s).astype(np.uint8),
                "tolerance": 0.0,  # Exact match expected for integers
            },
            {
                "dtype": np.uint8,
                "shape": (15, 8),
                "data": lambda s: np.random.randint(0, 255, s).astype(np.uint8),
                "tolerance": 0.0,
            },
            # Edge cases - single column
            {
                "dtype": np.float64,
                "shape": (1, 1),
                "data": lambda s: np.random.random(s).astype(np.float64),
                "tolerance": 1e-9,
            },
            # Edge cases - single row
            {
                "dtype": np.float64,
                "shape": (1, 10),
                "data": lambda s: np.random.random(s).astype(np.float64),
                "tolerance": 1e-9,
            },
            # Edge cases - zero values
            {
                "dtype": np.float64,
                "shape": (10, 10),
                "data": lambda s: np.zeros(s, dtype=np.float64),
                "tolerance": 0.0,
            },
            # Edge cases - negative values (for signed types)
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

    @timeout(60)
    def test_frame_python_to_java(self):
        """Test converting pandas DataFrame to SystemDS FrameBlock and writing to CSV."""
        combinations = [
            # Float32 column
            {"float32_col": np.random.random(50).astype(np.float32)},
            # Float64 column
            {"float64_col": np.random.random(50).astype(np.float64)},
            # Int32 column
            {"int32_col": np.random.randint(-1000, 1000, 50).astype(np.int32)},
            # Int64 column
            {"int64_col": np.random.randint(-1000000, 1000000, 50).astype(np.int64)},
            # Uint8 column
            {"uint8_col": np.random.randint(0, 255, 50).astype(np.uint8)},
            # All numeric types together
            {
                "float32_col": np.random.random(30).astype(np.float32),
                "float64_col": np.random.random(30).astype(np.float64),
                "int32_col": np.random.randint(-1000, 1000, 30).astype(np.int32),
                "int64_col": np.random.randint(-1000000, 1000000, 30).astype(np.int64),
                "uint8_col": np.random.randint(0, 255, 30).astype(np.uint8),
            },
            # Mixed numeric types with strings
            {
                "float32_col": np.random.random(25).astype(np.float32),
                "float64_col": np.random.random(25).astype(np.float64),
                "int32_col": np.random.randint(-1000, 1000, 25).astype(np.int32),
                "int64_col": np.random.randint(-1000000, 1000000, 25).astype(np.int64),
                "uint8_col": np.random.randint(0, 255, 25).astype(np.uint8),
                "string_col": [f"string_{i}" for i in range(25)],
            },
        ]

        for frame_dict in combinations:
            frame = pd.DataFrame(frame_dict)
            # Transfer into SystemDS and write to CSV
            frame_sds = self.sds.from_pandas(frame)
            frame_sds.write(
                self.temp_dir + "into_systemds_frame.csv", format="csv", header=False
            ).compute()

            # Read the CSV file using pandas
            result_df = pd.read_csv(
                self.temp_dir + "into_systemds_frame.csv", header=None
            )

            # For numeric columns, verify with allclose for floats, exact match for integers
            # For string columns, verify exact match
            for col_idx, col_name in enumerate(frame.columns):
                original_col = frame[col_name]
                result_col = result_df.iloc[:, col_idx]

                if pd.api.types.is_numeric_dtype(original_col):
                    original_dtype = original_col.dtype
                    # For integer types (int32, int64, uint8), use exact equality
                    if original_dtype in [np.int32, np.int64, np.uint8]:
                        self.assertTrue(
                            np.array_equal(
                                original_col.values.astype(original_dtype),
                                result_col.values.astype(original_dtype),
                            ),
                            f"Column {col_name} (dtype: {original_dtype}) integer values don't match exactly",
                        )
                    else:
                        # For float types (float32, float64), use allclose
                        self.assertTrue(
                            np.allclose(
                                original_col.values.astype(float),
                                result_col.values.astype(float),
                                equal_nan=True,
                            ),
                            f"Column {col_name} (dtype: {original_dtype}) float values don't match",
                        )
                else:
                    # For string columns, compare as strings
                    self.assertTrue(
                        (
                            original_col.astype(str).values
                            == result_col.astype(str).values
                        ).all(),
                        f"Column {col_name} string values don't match",
                    )

    @timeout(60)
    def test_frame_java_to_python(self):
        """Test reading CSV into SystemDS FrameBlock and converting back to pandas DataFrame."""
        combinations = [
            {"float32_col": np.random.random(50).astype(np.float32)},
            {"float64_col": np.random.random(50).astype(np.float64)},
            {"int32_col": np.random.randint(-1000, 1000, 50).astype(np.int32)},
            {"int64_col": np.random.randint(-1000000, 1000000, 50).astype(np.int64)},
            {"uint8_col": np.random.randint(0, 255, 50).astype(np.uint8)},
            # String column only
            {"text_col": [f"text_value_{i}" for i in range(30)]},
            # All numeric types together
            {
                "float32_col": np.random.random(30).astype(np.float32),
                "float64_col": np.random.random(30).astype(np.float64),
                "int32_col": np.random.randint(-1000, 1000, 30).astype(np.int32),
                "int64_col": np.random.randint(-1000000, 1000000, 30).astype(np.int64),
                "uint8_col": np.random.randint(0, 255, 30).astype(np.uint8),
            },
            # Mixed numeric types with strings
            {
                "float32_col": np.random.random(25).astype(np.float32),
                "float64_col": np.random.random(25).astype(np.float64),
                "int32_col": np.random.randint(-1000, 1000, 25).astype(np.int32),
                "int64_col": np.random.randint(-1000000, 1000000, 25).astype(np.int64),
                "uint8_col": np.random.randint(0, 255, 25).astype(np.uint8),
                "string_col": [f"string_{i}" for i in range(25)],
            },
        ]
        print("Running frame conversion test\n\n!!!!")
        for frame_dict in combinations:
            frame = pd.DataFrame(frame_dict)
            # Create a CSV file to read into SystemDS
            frame_sds = self.sds.from_pandas(frame)
            frame_sds = frame_sds.rbind(frame_sds)
            frame_out = frame_sds.compute()

            frame = pd.concat([frame, frame], ignore_index=True)

            # Verify it's a DataFrame
            self.assertIsInstance(frame_out, pd.DataFrame)

            # Verify shape matches
            self.assertEqual(frame.shape, frame_out.shape, "Frame shapes don't match")

            # Verify column data
            for col_name in frame.columns:
                original_col = frame[col_name]
                # FrameBlock to pandas may not preserve column names, so compare by position
                col_idx = list(frame.columns).index(col_name)
                result_col = frame_out.iloc[:, col_idx]

                if pd.api.types.is_numeric_dtype(original_col):
                    original_dtype = original_col.dtype
                    # For integer types (int32, int64, uint8), use exact equality
                    if original_dtype in [np.int32, np.int64, np.uint8]:
                        self.assertTrue(
                            np.array_equal(
                                original_col.values.astype(original_dtype),
                                result_col.values.astype(original_dtype),
                            ),
                            f"Column {col_name} (dtype: {original_dtype}) integer values don't match exactly",
                        )
                    else:
                        # For float types (float32, float64), use allclose
                        # print difference in case of failure
                        if not np.allclose(
                            original_col.values.astype(float),
                            result_col.values.astype(float),
                            equal_nan=True,
                            atol=1e-6,
                        ):
                            print(
                                f"Column {col_name} (dtype: {original_dtype}) float values don't match: {np.abs(original_col.values.astype(float) - result_col.values.astype(float))}"
                            )
                            self.assertTrue(
                                False,
                                f"Column {col_name} (dtype: {original_dtype}) float values don't match",
                            )

                else:
                    # For string columns, compare as strings
                    original_str = original_col.astype(str).values
                    result_str = result_col.astype(str).values
                    self.assertTrue(
                        (original_str == result_str).all(),
                        f"Column {col_name} string values don't match",
                    )

    @timeout(60)
    def test_frame_string_with_nulls(self):
        """Test converting pandas DataFrame with null string values."""
        # Create a simple DataFrame with 5 string values, 2 of them None
        df = pd.DataFrame({"string_col": ["hello", None, "world", None, "test"]})

        # Transfer into SystemDS and back
        frame_sds = self.sds.from_pandas(df)
        frame_sds = frame_sds.rbind(frame_sds)
        frame_out = frame_sds.compute()
        df = pd.concat([df, df], ignore_index=True)

        # Verify it's a DataFrame
        self.assertIsInstance(frame_out, pd.DataFrame)

        # Verify shape matches
        self.assertEqual(df.shape, frame_out.shape, "Frame shapes don't match")

        # Verify column data - check that None values are preserved
        original_col = df["string_col"]
        result_col = frame_out.iloc[:, 0]

        # Check each value
        for i in range(len(original_col)):
            original_val = original_col.iloc[i]
            result_val = result_col.iloc[i]

            if pd.isna(original_val):
                # Original is null, result should also be null
                self.assertTrue(
                    pd.isna(result_val),
                    f"Row {i}: Expected null but got '{result_val}'",
                )
            else:
                # Original is not null, result should match
                self.assertEqual(
                    str(original_val),
                    str(result_val),
                    f"Row {i}: Expected '{original_val}' but got '{result_val}'",
                )


if __name__ == "__main__":
    unittest.main(exit=False)
