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

import tempfile
import unittest

from systemds.scuro.drsearch.algebra import ALL_LAWS, AlgebraConfig, canonical_id
from systemds.scuro.drsearch.multimodal_optimizer import MultimodalOptimizer
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.sum import Sum
from tests.scuro.test_multimodal_ga_optimizer import (
    _FakeModality,
    _FakeUnimodalResults,
    _make_task,
)

PERMUTATIONS = AlgebraConfig(exhaustive_dedup_permutations=True)
CANONICAL = AlgebraConfig(
    laws=ALL_LAWS,
    feature_order_invariant=True,
    exhaustive_dedup_permutations=True,
    exhaustive_canonical=True,
)


def _fusion_dags(algebra, n=3):
    ids = [f"m{i}" for i in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        optimizer = MultimodalOptimizer(
            [_FakeModality(m) for m in ids],
            _FakeUnimodalResults({m: [object()] for m in ids}),
            [_make_task()],
            debug=False,
            checkpoint_dir=tmp,
            algebra=algebra,
        )
    optimizer.fusion_operators = [Concatenation, Sum, Average]
    dags = list(optimizer._generate_fusion_dags(ids, {m: 0 for m in ids}))
    return dags, optimizer.algebra_stats


class TestExhaustiveDedup(unittest.TestCase):
    def test_counts(self):
        legacy, _ = _fusion_dags(AlgebraConfig())
        ordered, stats = _fusion_dags(PERMUTATIONS)
        self.assertEqual((len(legacy), len(ordered)), (648, 108))
        self.assertEqual(stats["exhaustive_generated"], 648)
        self.assertEqual(stats["exhaustive_after_permutation_dedup"], 108)

    def test_canonical_matches_brute_force(self):
        ordered, _ = _fusion_dags(PERMUTATIONS)
        ids = {canonical_id(d, ALL_LAWS, True) for d in ordered}
        canonical, stats = _fusion_dags(CANONICAL)
        self.assertEqual(
            sorted(canonical_id(d, ALL_LAWS, True) for d in canonical), sorted(ids)
        )
        self.assertEqual(stats["exhaustive_after_canonical"], len(ids))
