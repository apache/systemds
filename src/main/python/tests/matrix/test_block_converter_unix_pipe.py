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


class TestMatrixBlockConverterUnixPipe(unittest.TestCase):

    sds: SystemDSContext = None
    temp_dir: str = "tests/iotests/temp_write_csv/"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(
            data_transfer_mode=1, logging_level=50, capture_stdout=True
        )
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

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

    def test_java_to_python(self):
        combinations = [  # (n_rows, n_cols)
            (5, 1),
            (10, 10),
        ]

        for n_rows, n_cols in combinations:
            matrix = np.random.random((n_rows, n_cols))

            # Create a CSV file to read into SystemDS
            pd.DataFrame(matrix).to_csv(
                self.temp_dir + "out_of_systemds_matrix.csv", header=False, index=False
            )

            matrix_sds = self.sds.read(
                self.temp_dir + "out_of_systemds_matrix.csv",
                data_type="matrix",
                format="csv",
            )
            matrix_out = matrix_sds.compute()

            # Verify the data
            self.assertTrue(np.allclose(matrix_out, matrix))


if __name__ == "__main__":
    unittest.main(exit=False)
