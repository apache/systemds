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

from systemds.scuro.drsearch.algebra import (
    ALL_LAWS,
    _brute_force_canonical,
    count_fusion_plans,
    count_search_space,
)
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.hadamard import Hadamard
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.representations.max import RowMax
from systemds.scuro.representations.sum import Sum

COMM = frozenset({"comm", "concat_perm"})
COMM_ASSOC = [Sum, Hadamard, RowMax]
MIXED = [Sum, Average, Concatenation, LSTM]


class TestPlanCounts(unittest.TestCase):
    def test_closed_forms_match_brute_force(self):
        for ops in (COMM_ASSOC, MIXED):
            for n in (2, 3, 4):
                with self.subTest(ops=[op.__name__ for op in ops], n=n):
                    counts = count_fusion_plans(n, ops, ALL_LAWS, True)
                    for key, laws in (
                        ("ordered", frozenset()),
                        ("commutative", COMM),
                        ("canonical", ALL_LAWS),
                    ):
                        brute = _brute_force_canonical(n, ops, laws, True)
                        self.assertEqual(counts[key], brute, key)

    def test_known_values(self):
        counts = count_fusion_plans(3, [Concatenation, Sum, Average], ALL_LAWS, True)
        self.assertEqual((counts["legacy_exhaustive"], counts["ordered"]), (648, 108))
        self.assertEqual(
            count_fusion_plans(4, COMM_ASSOC, COMM, True)["commutative"], 3240 // 8
        )
        self.assertIsNone(count_fusion_plans(5, MIXED, ALL_LAWS, True)["canonical"])
        self.assertEqual(
            count_fusion_plans(3, [Concatenation], COMM, True)["commutative"], 6
        )

    def test_search_space_sums_over_subsets(self):
        per_size = {n: count_fusion_plans(n, MIXED, ALL_LAWS, False) for n in (2, 3)}
        total = count_search_space([2, 2, 3], MIXED, ALL_LAWS, False)
        for key, value in total.items():
            self.assertEqual(
                value, (4 + 6 + 6) * per_size[2][key] + 12 * per_size[3][key], key
            )
