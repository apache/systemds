import logging
import unittest

import numpy as np

from systemds.context import SystemDSContext

from systemds.operator.nn_nodes.affine import affine

dim = 6
n = 10
m = 5
np.random.seed(1)
X = np.random.rand(n, dim)
W = np.random.rand(dim, m)
b = np.random.rand(m)
b.shape = (1, m)

class TestAffine(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_affine(self):
        Xm = self.sds.from_numpy(X)
        Wm = self.sds.from_numpy(W)
        bm = self.sds.from_numpy(b)
        out = affine(Xm, Wm, bm).compute()
        expected = np.matmul(X, W) + b
        print(out)
        eval = np.allclose(out, expected)
        self.assertEqual(eval, True, "Failed")



if __name__ == '__main__':
    unittest.main()
