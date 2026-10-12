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

import itertools
import unittest

import numpy as np

from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.representations.representation import RepresentationStats
from tests.scuro.test_algebra_laws import _fuse

# Other test modules register fusion operators of their own.
OPERATORS = [
    op
    for op in Registry().get_fusion_operators()
    if op.__module__.startswith("systemds.") and not op().needs_training
]
OPERANDS = {
    "equal": [(5, 6), (5, 6)],
    "unequal": [(5, 6), (5, 3)],
    "sequence": [(5, 4, 6), (5, 3)],
}


class TestPredictedStats(unittest.TestCase):
    def test_predicted_stats_match_execution(self):
        for op, case in itertools.product(OPERATORS, OPERANDS):
            with self.subTest(op=op.__name__, case=case):
                rng = np.random.default_rng(0)
                operands = [
                    rng.standard_normal(shape).astype(np.float32)
                    for shape in OPERANDS[case]
                ]
                stats = [RepresentationStats(len(x), x.shape[1:]) for x in operands]
                fused = _fuse(op, {}, *operands)
                self.assertEqual(
                    tuple(op().get_output_stats(stats).output_shape), fused.shape[1:]
                )
                self.assertGreater(
                    op().estimate_peak_memory_bytes(stats)["cpu_peak_bytes"],
                    fused.nbytes,
                )


if __name__ == "__main__":
    unittest.main()
