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

from systemds.scuro.drsearch.operator_registry import Registry
from tests.scuro.data_generator import ModalityRandomDataGenerator
from systemds.scuro.modality.type import ModalityType

# built-in fusions only: other test modules register their own, and a fusion
# that needs training cannot be combined without labels
FUSION_OPERATORS = [
    op
    for op in Registry().get_fusion_operators()
    if op.__module__.startswith("systemds.") and not op().needs_training
]


class TestFusionOrders(unittest.TestCase):
    """
    Tests the order properties of the fusion operators against the flags they
    declare. Swapping the inputs of a pair, or the first pair of a chain, only
    keeps the result for a commutative operator. A pairwise chain only matches
    the n-ary call for an associative one.
    """

    @classmethod
    def setUpClass(cls):
        # These properties do not depend on the input shape, so the data can
        # be small.
        cls.num_instances = 4
        cls.num_features = 8
        cls.data_generator = ModalityRandomDataGenerator()

    def setUp(self):
        self.r_1 = self.data_generator.create1DModality(
            self.num_instances, self.num_features, ModalityType.AUDIO
        )
        self.r_2 = self.data_generator.create1DModality(
            self.num_instances, self.num_features, ModalityType.TEXT
        )
        self.r_3 = self.data_generator.create1DModality(
            self.num_instances, self.num_features, ModalityType.TEXT
        )

    @staticmethod
    def _equal(left, right):
        return np.array_equal(np.asarray(left.data), np.asarray(right.data))

    def test_fusion_order_properties(self):
        for fusion_operator in FUSION_OPERATORS:
            name = fusion_operator.__name__
            commutative = fusion_operator().commutative
            associative = fusion_operator().associative
            with self.subTest(fusion=name):
                r_1_r_2 = self.r_1.combine(self.r_2, fusion_operator())
                r_2_r_1 = self.r_2.combine(self.r_1, fusion_operator())
                r_1_r_2_r_3 = r_1_r_2.combine(self.r_3, fusion_operator())
                r_2_r_1_r_3 = r_2_r_1.combine(self.r_3, fusion_operator())
                r1_r2_r3 = self.r_1.combine([self.r_2, self.r_3], fusion_operator())

                self.assertEqual(
                    self._equal(r_1_r_2, r_2_r_1),
                    commutative,
                    f"{name}: swapping the two inputs",
                )
                self.assertEqual(
                    self._equal(r_1_r_2_r_3, r_2_r_1_r_3),
                    commutative,
                    f"{name}: swapping the first pair of a chain",
                )
                self.assertEqual(
                    self._equal(r_1_r_2_r_3, r1_r2_r3),
                    associative,
                    f"{name}: pairwise chain against the n-ary call",
                )
                self.assertFalse(
                    self._equal(r_1_r_2, r1_r2_r3),
                    f"{name}: a pair against all three inputs",
                )
