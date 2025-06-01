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

import numpy as np
import pandas as pd
from systemds.context import SystemDSContext


class TestReadCSV(unittest.TestCase):
    sds: SystemDSContext = None
    temp_dir: str = "tests/iotests/temp_write_csv/"
    n_cols = 3
    n_rows = 100

    df = pd.DataFrame(
        {
            "col1": [f"ss{i}s.{i}" for i in range(n_rows)],
            "col2": [i for i in range(n_rows)],
            "col3": [i * 0.1 for i in range(n_rows)],
        }
    )

    df2 = pd.DataFrame(
        {
            "col2": [i for i in range(n_rows)],
            "col3": [i * 0.1 for i in range(n_rows)],
        }
    )

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(logging_level=50)
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_write_read_data_frame_csv_header(self):
        filename = self.temp_dir + "data_frame_header.csv"
        self.df.to_csv(filename, index=False, header=True)
        result_df = self.sds.read(filename, data_type="frame").compute()
        self.compare_frame(result_df, self.df)

    def test_write_read_data_frame_csv_header_active(self):
        filename = self.temp_dir + "data_frame_header_active.csv"
        self.df.to_csv(filename, index=False, header=True)
        result_df = self.sds.read(filename, data_type="frame", header=True).compute()
        self.compare_frame(result_df, self.df)

    def test_write_read_data_frame_csv_no_header(self):
        filename = self.temp_dir + "data_frame_no_header.csv"
        self.df.to_csv(filename, index=False, header=False)
        result_df = self.sds.read(filename, data_type="frame", header=False).compute()
        self.compare_frame(result_df, self.df)

    def test_write_read_matrix_csv_no_extra_argument(self):
        filename = self.temp_dir + "data_matrix_no_header.csv"
        self.df2.to_csv(filename, index=False, header=False)
        result_df = (self.sds.read(filename)).compute()
        self.assertTrue(np.allclose(self.df2.to_numpy(), result_df))

    def test_write_read_matrix_csv_no_extra_argument_header(self):
        filename = self.temp_dir + "data_matrix_header.csv"
        self.df2.to_csv(filename, index=False, header=True)
        result_df = (self.sds.read(filename, header=True)).compute()
        self.assertTrue(np.allclose(self.df2.to_numpy(), result_df))

    def test_write_read_matrix_csv_no_extra_argument_header_csv(self):
        filename = self.temp_dir + "data_matrix_header_2.csv"
        self.df2.to_csv(filename, index=False, header=True)
        result_df = (self.sds.read(filename, format="csv", header=True)).compute()
        self.assertTrue(np.allclose(self.df2.to_numpy(), result_df))

    def compare_frame(self, a: pd.DataFrame, b: pd.DataFrame):
        a = a.astype(str)
        b = b.astype(str)
        self.assertTrue(isinstance(a, pd.DataFrame))
        self.assertTrue(isinstance(b, pd.DataFrame))
        self.assertTrue((a.values == b.values).all())


if __name__ == "__main__":
    unittest.main(exit=False)
