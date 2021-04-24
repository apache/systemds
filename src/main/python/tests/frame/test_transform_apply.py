import os
import shutil
import sys
import unittest

import pandas as pd
import numpy as np
import json
from systemds.context import SystemDSContext
from systemds.frame import Frame
from systemds.matrix import Matrix


class TestTransformApply(unittest.TestCase):

    sds: SystemDSContext = None
    HOMES_PATH = "tests/frame/data/homes.csv"
    HOMES_SCHEMA = '"int,string,int,int,double,int,boolean,int,int"'

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def tearDown(self):
        pass

    def test_apply_recode_bin(self):
        JSPEC_PATH = "tests/frame/data/homes.tfspec_bin2.json"
        with open(JSPEC_PATH) as jspec_file:
            JSPEC = json.load(jspec_file)
        F1 = self.sds.read(
            self.HOMES_PATH,
            data_type="frame",
            schema=self.HOMES_SCHEMA,
            format="csv",
            header=True,
        )
        pd_F1 = F1.compute()
        jspec = self.sds.read(JSPEC_PATH, data_type="scalar", value_type="string")
        X, M = F1.transform_encode(spec=jspec).compute()
        self.assertTrue(isinstance(X, np.ndarray))
        self.assertTrue(isinstance(M, pd.DataFrame))
        self.assertTrue(X.shape == pd_F1.shape)
        self.assertTrue(np.all(np.isreal(X)))
        relevant_columns = set()
        for col_name in JSPEC["recode"]:
            relevant_columns.add(pd_F1.columns.get_loc(col_name))
            self.assertTrue(M[col_name].nunique() == pd_F1[col_name].nunique())
        for binning in JSPEC["bin"]:
            col_name = binning["name"]
            relevant_columns.add(pd_F1.columns.get_loc(col_name))
            self.assertTrue(M[col_name].nunique() == binning["numbins"])

        X2 = F1.transform_apply(spec=jspec, meta=Frame(self.sds, M)).compute()
        self.assertTrue(X.shape == X2.shape)
        self.assertTrue(np.all(np.isreal(X2)))



if __name__ == "__main__":
    unittest.main(exit=False)
