import unittest

import numpy as np

from systemds.context import SystemDSContext

from systemds.operator.nn_nodes.affine import Affine

dim = 6
n = 10
m = 5
np.random.seed(11)
X = np.random.rand(n, dim)

np.random.seed(10)
W = np.random.rand(dim, m)
b = np.random.rand(m)


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

        affine = Affine(dim, m, 10)
        out = affine.forward(Xm)
        print(out.compute())
        print(out.script_str)
        dout = self.sds.from_numpy(np.random.rand(n, m))
        dX, dW, db = affine.backward(dout)
        assert True


if __name__ == '__main__':
    unittest.main()
