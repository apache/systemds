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

import pandas as pd
from systemds.context import SystemDSContext


class TestRCBind(unittest.TestCase):

    sds: SystemDSContext = None

    # shape (2, 3)
    df_cb_1 = pd.DataFrame(
        {"col1": ["col1_hello", "col1_world"], "col2": [0, 1], "col3": [0.0, 0.1]}
    )
    # shape (2, 2)
    df_cb_2 = pd.DataFrame({"col4": ["col4_hello", "col4_world"], "col5": [0, 1]})
    df_cb_3 = pd.DataFrame({"col6": ["col6_hello", "col6_world"], "col7": [0, 1]})

    # shape (2, 3)
    df_rb_1 = pd.DataFrame(
        {"col1": ["col1_hello_1", "col1_world_1"], "col2": [0, 1], "col3": [0.0, 0.1]}
    )
    # shape (4, 3)
    df_rb_2 = pd.DataFrame(
        {
            "col1": ["col1_hello_2", "col1_world_2", "col1_hello_2", "col1_world_2"],
            "col2": [2, 3, 4, 5],
            "col3": [0.2, 0.3, 0.4, 0.5],
        }
    )
    # shape (3, 3)
    df_rb_3 = pd.DataFrame(
        {
            "col1": ["col1_hello_3", "col1_world_3", "col1_hello_3"],
            "col2": [6, 7, 8],
            "col3": [0.6, 0.7, 0.8],
        }
    )

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_r_bind_pair(self):
        f1 = self.sds.from_pandas(self.df_rb_1)
        f2 = self.sds.from_pandas(self.df_rb_2)
        result_df = f1.rbind(f2).compute()
        self.assertTrue(isinstance(result_df, pd.DataFrame))
        target_df = pd.concat([self.df_rb_1, self.df_rb_2], ignore_index=True)
        self.assertTrue(target_df.equals(result_df))

    def test_r_bind_triple(self):
        f1 = self.sds.from_pandas(self.df_rb_1)
        f2 = self.sds.from_pandas(self.df_rb_2)
        f3 = self.sds.from_pandas(self.df_rb_3)
        result_df = f1.rbind(f2).rbind(f3).compute()
        self.assertTrue(isinstance(result_df, pd.DataFrame))
        target_df = pd.concat(
            [self.df_rb_1, self.df_rb_2, self.df_rb_3], ignore_index=True
        )
        self.assertTrue(target_df.equals(result_df))

    def test_r_bind_triple_twostep(self):
        f1 = self.sds.from_pandas(self.df_rb_1)
        f2 = self.sds.from_pandas(self.df_rb_2)
        f3 = self.sds.from_pandas(self.df_rb_3)
        tmp_df = f1.rbind(f2).compute()
        result_df = self.sds.from_pandas(tmp_df).rbind(f3).compute()
        self.assertTrue(isinstance(result_df, pd.DataFrame))
        target_df = pd.concat(
            [self.df_rb_1, self.df_rb_2, self.df_rb_3], ignore_index=True
        )
        self.assertTrue(target_df.equals(result_df))

    def test_c_bind_pair(self):
        f1 = self.sds.from_pandas(self.df_cb_1)
        f2 = self.sds.from_pandas(self.df_cb_2)
        result_df = f1.cbind(f2).compute()
        self.assertTrue(isinstance(result_df, pd.DataFrame))
        target_df = pd.concat([self.df_cb_1, self.df_cb_2], axis=1)
        self.assertTrue(target_df.equals(result_df))

    def test_c_bind_triple(self):
        f1 = self.sds.from_pandas(self.df_cb_1)
        f2 = self.sds.from_pandas(self.df_cb_2)
        f3 = self.sds.from_pandas(self.df_cb_3)
        result_df = f1.cbind(f2).cbind(f3).compute()
        self.assertTrue(isinstance(result_df, pd.DataFrame))
        target_df = pd.concat([self.df_cb_1, self.df_cb_2, self.df_cb_3], axis=1)
        self.assertTrue(target_df.equals(result_df))

    def test_c_bind_triple_twostep(self):
        f1 = self.sds.from_pandas(self.df_cb_1)
        f2 = self.sds.from_pandas(self.df_cb_2)
        f3 = self.sds.from_pandas(self.df_cb_3)
        tmp_df = f1.cbind(f2).compute()
        result_df = self.sds.from_pandas(tmp_df).cbind(f3).compute()
        self.assertTrue(isinstance(result_df, pd.DataFrame))
        target_df = pd.concat([self.df_cb_1, self.df_cb_2, self.df_cb_3], axis=1)
        self.assertTrue(target_df.equals(result_df))


if __name__ == "__main__":
    unittest.main(exit=False)
