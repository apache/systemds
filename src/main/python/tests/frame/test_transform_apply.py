import os
import shutil
import sys
import unittest

import pandas as pd
import numpy as np
from systemds.context import SystemDSContext
from systemds.frame import Frame
from systemds.matrix import Matrix


class TestTransformApply(unittest.TestCase):

    sds: SystemDSContext = None
    HOMES_PATH = "tests/frame/data/homes.csv"
    HOMES_SCHEMA = '"int,string,int,int,double,int,boolean,int,int"'
    JSPEC_PATH = "tests/frame/data/homes.tfspec_bin2.json"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def tearDown(self):
        pass

    def test_apply(self):
        F1 = self.sds.read(
            self.HOMES_PATH,
            data_type="frame",
            schema=self.HOMES_SCHEMA,
            format="csv",
            header=True,
        )
        jspec = self.sds.read(self.JSPEC_PATH, data_type="scalar", value_type="string")
        X, M = F1.transform_encode(spec=jspec).compute(verbose=True)
        print(X)
        print(M)
        X2 = F1.transform_apply(spec=jspec, meta=Frame(self.sds, M)).compute(
            verbose=True
        )
        print(X2)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(exit=False)
