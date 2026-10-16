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
from types import SimpleNamespace

from systemds.scuro.drsearch.algebra import AlgebraConfig
from systemds.scuro.drsearch.unimodal_optimizer import UnimodalOptimizer
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.sum import Sum

LEGACY = AlgebraConfig()
LAW_AWARE = AlgebraConfig(cse_law_aware=True)
FULL = AlgebraConfig.full()


def _builder(algebra):
    optimizer = UnimodalOptimizer(
        [SimpleNamespace(modality_id=0)],
        [SimpleNamespace(model=SimpleNamespace(name="task"), expected_dim=1)],
        debug=False,
        window_combination_chains=2,
        algebra=algebra,
    )
    builder = optimizer.builders[0]
    return optimizer, builder, [builder.create_leaf_node(0, i) for i in range(2)]


def _nodes(builder, op):
    return [n for n in builder.global_nodes if n.operation is op]


class TestLawAwareCSE(unittest.TestCase):
    def test_window_concatenation_rotations(self):
        for algebra, expected in ((LEGACY, 1), (LAW_AWARE, 2), (FULL, 2)):
            with self.subTest(algebra=algebra):
                optimizer, builder, windows = _builder(algebra)
                optimizer.combine_windowed_representations(builder, windows)
                self.assertEqual(len(_nodes(builder, Concatenation)), expected)
                prevented = builder.algebra_stats["cse_unsafe_merges_prevented"]
                self.assertEqual(prevented, int(algebra.cse_law_aware))

    def test_commutative_fusion_merges_in_both_modes(self):
        cases = [
            (Sum, {}, LEGACY, 1),
            (Sum, {}, LAW_AWARE, 1),
            (Concatenation, {"preserve_leading_axis": False}, LEGACY, 1),
            (Concatenation, {"preserve_leading_axis": False}, LAW_AWARE, 2),
            (Concatenation, {"preserve_leading_axis": False}, FULL, 1),
        ]
        for op, params, algebra, expected in cases:
            with self.subTest(op=op.__name__, algebra=algebra):
                _, builder, (a, b) = _builder(algebra)
                builder.create_operation_node(op, [a, b], params)
                builder.create_operation_node(op, [b, a], params)
                self.assertEqual(len(_nodes(builder, op)), expected)
                merged_by_law = builder.algebra_stats["cse_merges_by_law"]
                self.assertEqual(
                    merged_by_law, int(algebra.cse_law_aware and expected == 1)
                )


if __name__ == "__main__":
    unittest.main()
