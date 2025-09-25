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

import numpy as np
from numpy.testing import assert_almost_equal

from systemds.operator.nn.affine import Affine
from systemds.operator.nn.relu import ReLU
from systemds.operator.nn.sequential import Sequential
from systemds.operator import Matrix, MultiReturn
from systemds.operator.nn.layer import Layer
from systemds.context import SystemDSContext


class TestLayerImpl(Layer):
    def __init__(self, test_id):
        super().__init__()
        self.test_id = test_id

    def _instance_forward(self, X: Matrix):
        return X + self.test_id

    def _instance_backward(self, dout: Matrix, X: Matrix):
        return dout - self.test_id


class MultiReturnImpl(Layer):
    def __init__(self, sds):
        super().__init__()
        self.sds = sds

    def _instance_forward(self, X: Matrix):
        return MultiReturn(self.sds, "test.dml", output_nodes=[X, "some_random_return"])

    def _instance_backward(self, dout: Matrix, X: Matrix):
        return MultiReturn(
            self.sds, "test.dml", output_nodes=[dout, X, "some_random_return"]
        )


class TestSequential(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_init_with_multiple_args(self):
        """
        Test that Sequential is correctly initialized if multiple layers are passed as arguments
        """
        model = Sequential(TestLayerImpl(1), TestLayerImpl(2), TestLayerImpl(3))
        self.assertEqual(len(model.layers), 3)
        self.assertEqual(model.layers[0].test_id, 1)
        self.assertEqual(model.layers[1].test_id, 2)
        self.assertEqual(model.layers[2].test_id, 3)

    def test_init_with_list(self):
        """
        Test that Sequential is correctly initialized if list of layers is passed as argument
        """
        model = Sequential([TestLayerImpl(1), TestLayerImpl(2), TestLayerImpl(3)])
        self.assertEqual(len(model.layers), 3)
        self.assertEqual(model.layers[0].test_id, 1)
        self.assertEqual(model.layers[1].test_id, 2)
        self.assertEqual(model.layers[2].test_id, 3)

    def test_len(self):
        """
        Test that len() returns the number of layers
        """
        model = Sequential([TestLayerImpl(1), TestLayerImpl(2), TestLayerImpl(3)])
        self.assertEqual(len(model), 3)

    def test_getitem(self):
        """
        Test that Sequential[index] returns the layer at the given index
        """
        model = Sequential([TestLayerImpl(1), TestLayerImpl(2), TestLayerImpl(3)])
        self.assertEqual(model[1].test_id, 2)

    def test_setitem(self):
        """
        Test that Sequential[index] = layer sets the layer at the given index
        """
        model = Sequential([TestLayerImpl(1), TestLayerImpl(2), TestLayerImpl(3)])
        model[1] = TestLayerImpl(4)
        self.assertEqual(model[1].test_id, 4)

    def test_delitem(self):
        """
        Test that del Sequential[index] removes the layer at the given index
        """
        model = Sequential([TestLayerImpl(1), TestLayerImpl(2), TestLayerImpl(3)])
        del model[1]
        self.assertEqual(len(model.layers), 2)
        self.assertEqual(model[1].test_id, 3)

    def test_iter(self):
        """
        Test that iter() returns an iterator over the layers
        """
        model = Sequential([TestLayerImpl(1), TestLayerImpl(2), TestLayerImpl(3)])
        for i, layer in enumerate(model):
            self.assertEqual(layer.test_id, i + 1)

    def test_push(self):
        """
        Test that push() adds a layer
        """
        model = Sequential()
        model.push(TestLayerImpl(1))
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(model.layers[0].test_id, 1)

    def test_pop(self):
        """
        Test that pop() removes the last layer
        """
        model = Sequential([TestLayerImpl(1), TestLayerImpl(2), TestLayerImpl(3)])
        layer = model.pop()
        self.assertEqual(len(model.layers), 2)
        self.assertEqual(layer.test_id, 3)

    def test_reversed(self):
        """
        Test that reversed() returns an iterator over the layers in reverse order
        """
        model = Sequential([TestLayerImpl(1), TestLayerImpl(2), TestLayerImpl(3)])
        for i, layer in enumerate(reversed(model)):
            self.assertEqual(layer.test_id, 3 - i)

    def test_forward(self):
        """
        Test that forward() calls forward() on all layers
        """
        model = Sequential([TestLayerImpl(1), TestLayerImpl(2), TestLayerImpl(3)])
        in_matrix = self.sds.from_numpy(np.array([[1, 2], [3, 4]]))
        out_matrix = model.forward(in_matrix).compute()
        self.assertEqual(out_matrix.tolist(), [[7, 8], [9, 10]])

    def test_forward_actual_layers(self):
        """
        Test forward() with actual layers
        """
        params = [
            np.array([[0.5, -0.5], [-0.5, 0.5]]),
            np.array([[0.1, -0.1]]),
            np.array([[0.4, -0.4], [-0.4, 0.4]]),
            np.array([[0.2, -0.2]]),
            np.array([[0.3, -0.3], [-0.3, 0.3]]),
            np.array([[0.3, -0.3]]),
        ]

        model = Sequential(
            [
                Affine(self.sds, 2, 2),
                ReLU(self.sds),
                Affine(self.sds, 2, 2),
                ReLU(self.sds),
                Affine(self.sds, 2, 2),
            ]
        )

        for i, layer in enumerate(model):
            if isinstance(layer, Affine):
                layer.weight = self.sds.from_numpy(params[i])
                layer.bias = self.sds.from_numpy(params[i + 1])

        in_matrix = self.sds.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
        out_matrix = model.forward(in_matrix).compute()
        expected = np.array([[0.3120, -0.3120], [0.3120, -0.3120]])
        assert_almost_equal(out_matrix, expected)

    def test_backward_actual_layers(self):
        """
        Test backward() with actual layers
        """
        params = [
            np.array([[0.5, -0.5], [-0.5, 0.5]]),
            np.array([[0.1, -0.1]]),
            np.array([[0.4, -0.4], [-0.4, 0.4]]),
            np.array([[0.2, -0.2]]),
            np.array([[0.3, -0.3], [-0.3, 0.3]]),
            np.array([[0.3, -0.3]]),
        ]

        model = Sequential(
            [
                Affine(self.sds, 2, 2),
                ReLU(self.sds),
                Affine(self.sds, 2, 2),
                ReLU(self.sds),
                Affine(self.sds, 2, 2),
            ]
        )

        for i, layer in enumerate(model):
            if isinstance(layer, Affine):
                layer.weight = self.sds.from_numpy(params[i])
                layer.bias = self.sds.from_numpy(params[i + 1])

        in_matrix = self.sds.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
        out_matrix = model.forward(in_matrix)
        gradient = model.backward(out_matrix, in_matrix).compute()

        # Test returned gradient
        expected = np.array([[0.14976, -0.14976], [0.14976, -0.14976]])
        assert_almost_equal(gradient, expected)

        # Test if layers have been updated correctly
        expected_gradients = [
            np.array([[0.14976, -0.14976], [0.14976, -0.14976]]),
            np.array([[0.14976, -0.14976], [0.14976, -0.14976]]),
            np.array([[0.1872, -0.1872], [0.1872, -0.1872]]),
        ]
        for i, layer in enumerate(model):
            if isinstance(layer, Affine):
                assert_almost_equal(layer._X.compute(), expected_gradients[int(i / 2)])

    def test_multireturn_forward_pass(self):
        """
        Test that forward() handles MultiReturn correctly
        """
        model = Sequential(MultiReturnImpl(self.sds), TestLayerImpl(1))
        in_matrix = self.sds.from_numpy(np.array([[1, 2], [3, 4]]))
        out_matrix = model.forward(in_matrix).compute()
        self.assertEqual(out_matrix.tolist(), [[2, 3], [4, 5]])

    def test_multireturn_backward_pass(self):
        """
        Test that backward() handles MultiReturn correctly
        """
        model = Sequential(TestLayerImpl(1), MultiReturnImpl(self.sds))
        in_matrix = self.sds.from_numpy(np.array([[1, 2], [3, 4]]))
        out_matrix = self.sds.from_numpy(np.array([[2, 3], [4, 5]]))
        gradient = model.backward(out_matrix, in_matrix).compute()
        self.assertEqual(gradient.tolist(), [[1, 2], [3, 4]])

    def test_multireturn_variation_multiple(self):
        """
        Test that multiple MultiReturn after each other are handled correctly
        """
        model = Sequential(MultiReturnImpl(self.sds), MultiReturnImpl(self.sds))
        in_matrix = self.sds.from_numpy(np.array([[1, 2], [3, 4]]))
        out_matrix = model.forward(in_matrix).compute()
        self.assertEqual(out_matrix.tolist(), [[1, 2], [3, 4]])
        gradient = model.backward(self.sds.from_numpy(out_matrix), in_matrix).compute()
        self.assertEqual(gradient.tolist(), [[1, 2], [3, 4]])

    def test_multireturn_variation_single_to_multiple(self):
        """
        Test that a single return into multiple MultiReturn are handled correctly
        """
        model = Sequential(
            TestLayerImpl(1), MultiReturnImpl(self.sds), MultiReturnImpl(self.sds)
        )
        in_matrix = self.sds.from_numpy(np.array([[1, 2], [3, 4]]))
        out_matrix = model.forward(in_matrix).compute()
        self.assertEqual(out_matrix.tolist(), [[2, 3], [4, 5]])
        gradient = model.backward(self.sds.from_numpy(out_matrix), in_matrix).compute()
        self.assertEqual(gradient.tolist(), [[1, 2], [3, 4]])

    def test_multireturn_variation_multiple_to_single(self):
        """
        Test that multiple MultiReturn into a single return are handled correctly
        """
        model = Sequential(
            MultiReturnImpl(self.sds), MultiReturnImpl(self.sds), TestLayerImpl(1)
        )
        in_matrix = self.sds.from_numpy(np.array([[1, 2], [3, 4]]))
        out_matrix = model.forward(in_matrix).compute()
        self.assertEqual(out_matrix.tolist(), [[2, 3], [4, 5]])
        gradient = model.backward(self.sds.from_numpy(out_matrix), in_matrix).compute()
        self.assertEqual(gradient.tolist(), [[1, 2], [3, 4]])

    def test_multireturn_variation_sandwich(self):
        """
        Test that a single return between two MultiReturn are handled correctly
        """
        model = Sequential(
            MultiReturnImpl(self.sds), TestLayerImpl(1), MultiReturnImpl(self.sds)
        )
        in_matrix = self.sds.from_numpy(np.array([[1, 2], [3, 4]]))
        out_matrix = model.forward(in_matrix).compute()
        self.assertEqual(out_matrix.tolist(), [[2, 3], [4, 5]])
        gradient = model.backward(self.sds.from_numpy(out_matrix), in_matrix).compute()
        self.assertEqual(gradient.tolist(), [[1, 2], [3, 4]])
