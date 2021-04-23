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


class TestTransformEncode(unittest.TestCase):

    sds: SystemDSContext = None
    HOMES_PATH = "tests/frame/data/homes.csv"
    HOMES_SCHEMA = '"int,string,int,int,double,int,boolean,int,int"'
    JSPEC_PATH = "tests/frame/data/homes.tfspec_recode2.json"
    with open(JSPEC_PATH) as jspec_file:
        JSPEC = json.load(jspec_file)

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def tearDown(self):
        pass

    def test_encode(self):
        F1 = self.sds.read(
            self.HOMES_PATH,
            data_type="frame",
            schema=self.HOMES_SCHEMA,
            format="csv",
            header=True,
        )
        pd_F1 = F1.compute()
        jspec = self.sds.read(self.JSPEC_PATH, data_type="scalar", value_type="string")
        X, M = F1.transform_encode(spec=jspec).compute(verbose=True)
        self.assertTrue(X.shape == pd_F1.shape)
        self.assertTrue(np.all(np.isreal(X)))


if __name__ == "__main__":
    unittest.main(exit=False)
