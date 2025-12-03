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


class TestFrameConverterUnixPipe(unittest.TestCase):

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
