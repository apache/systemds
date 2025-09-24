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

import shutil
import unittest

import pandas as pd
from systemds.context import SystemDSContext


class TestWriteRead(unittest.TestCase):

    sds: SystemDSContext = None
    temp_dir: str = "tests/frame/temp_write/"
    n_cols = 3
    n_rows = 100
    df = pd.DataFrame(
        {
            "col1": [f"col1_string_{i}" for i in range(n_rows)],
            "col2": [i for i in range(n_rows)],
            "col3": [i * 0.1 for i in range(n_rows)],
        }
    )

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_read_binary(self):
        frame = self.sds.from_pandas(self.df)
        frame.write(self.temp_dir + "01").compute()
        NX = self.sds.read(self.temp_dir + "01", data_type="frame")
        result_df = NX.compute()
        self.assertTrue((self.df.values == result_df.values).all())

    def test_write_read_csv(self):
        frame = self.sds.from_pandas(self.df)
        frame.write(self.temp_dir + "02", header=True, format="csv").compute()
        NX = self.sds.read(self.temp_dir + "02", data_type="frame", format="csv")
        result_df = NX.compute()
        self.assertTrue(isinstance(result_df, pd.DataFrame))
        self.assertTrue(self.df.equals(result_df))


if __name__ == "__main__":
    unittest.main(exit=False)
