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

from systemds.context import SystemDSContext
from systemds.operator.nn.layer import Layer


class TestLayer(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_init(self):
        """
        Test that the source is created correctly from dml_script param when layer is initialized
        """
        _ = Layer(self.sds, "relu.dml")
        self.assertIsNotNone(Layer._source)
        self.assertTrue(Layer._source.operation.endswith('relu.dml"'))
        self.assertEqual(Layer._source._Source__name, "relu")

    def test_notimplemented(self):
        """
        Test that NotImplementedError is raised
        """

        class TestLayerImpl(Layer):
            pass

        layer = TestLayerImpl(self.sds, "relu.dml")
        with self.assertRaises(NotImplementedError):
            layer.forward(None)
        with self.assertRaises(NotImplementedError):
            layer.backward(None)
        with self.assertRaises(NotImplementedError):
            TestLayerImpl.forward(None)
        with self.assertRaises(NotImplementedError):
            TestLayerImpl.backward(None)

    def test_class_source_assignments(self):
        """
        Test that the source is not shared between interface and implementation class
        """

        class TestLayerImpl(Layer):
            @classmethod
            def _create_source(cls, sds_context: SystemDSContext, dml_script: str):
                cls._source = "test"

        _ = Layer(self.sds, "relu.dml")
        _ = TestLayerImpl(self.sds, "relu.dml")

        self.assertNotEqual(Layer._source, "test")
        self.assertEqual(TestLayerImpl._source, "test")
