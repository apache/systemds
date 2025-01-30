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
from systemds.context import SystemDSContext


def create_dataframe(n_rows, n_cols, mixed=True):
    return pd.DataFrame(
        {
            f"C{i+1}": [
                f"col{i+1}_string_{j}" if i == 0 and mixed else j + i
                for j in range(n_rows)
            ]
            for i in range(n_cols)
        }
    )


class TestPandasFromToSystemds(unittest.TestCase):

    sds: SystemDSContext = None
    temp_dir: str = "tests/iotests/temp_write_csv/"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_into_systemds(self):
        combinations = [  # (n_rows, n_cols, mixed)
            (3, 2, True),  # Test un-parallelized code (rows <= 4)
            (10, 5, True),  # Test parallelized column-wise code
            (5, 10, True),  # Test parallelized column-wise mixed code
            (5, 10, False),  # Test parallelized row-wise code
        ]

        for n_rows, n_cols, mixed in combinations:
            df = create_dataframe(n_rows, n_cols, mixed)

            # Transfer into SystemDS and write to CSV
            frame = self.sds.from_pandas(df)
            frame.write(
                self.temp_dir + "into_systemds.csv", format="csv", header=True
            ).compute(verbose=True)

            # Read the CSV file using pandas
            result_df = pd.read_csv(self.temp_dir + "into_systemds.csv")

            # Verify the data
            self.assertTrue(isinstance(result_df, pd.DataFrame))
            self.assertTrue(df.equals(result_df))

    def test_out_of_systemds(self):
        n_rows = 3
        n_cols = 2
        df = create_dataframe(n_rows, n_cols)

        # Create a CSV file to read into SystemDS
        df.to_csv(self.temp_dir + "out_of_systemds.csv", header=False, index=False)

        # Read the CSV file into SystemDS and then compute back to pandas
        frame = self.sds.read(
            self.temp_dir + "out_of_systemds.csv", data_type="frame", format="csv"
        )
        result_df = frame.replace("xyz", "yzx").compute()

        # Verify the data
        result_df["C2"] = result_df["C2"].astype(int)

        self.assertTrue(isinstance(result_df, pd.DataFrame))
        self.assertTrue(df.equals(result_df))


if __name__ == "__main__":
    unittest.main(exit=False)
