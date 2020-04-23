import unittest
import test_util
from onnx_systemds import onnx2systemds


class TestSimpleOperators(unittest.TestCase):
    def test_simple_mat_add(self):
        name = "simple_mat_add"
        test_util.run_and_compare_output(name, self)

    def test_simple_mat_add_mul_sub(self):
        name = "simple_mat_add_mul_sub"
        test_util.run_and_compare_output(name, self)

    def test_simple_mat_initialized(self):
        name = "simple_mat_initialized"
        test_util.run_and_compare_output(name, self)


if __name__ == '__main__':
    unittest.main()
