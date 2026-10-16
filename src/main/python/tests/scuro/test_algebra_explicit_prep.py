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
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

from systemds.scuro.drsearch.algebra import (
    ALL_LAWS,
    AlgebraConfig,
    canonical_id,
    count_explicit_prep,
    sample_stats,
    typecheck,
)
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.multimodal_ga_optimizer import (
    DagGenome,
    MultimodalDeapOptimizer,
)
from systemds.scuro.drsearch.multimodal_optimizer import MultimodalOptimizer
from systemds.scuro.drsearch.representation_dag import CSEAwareDAGBuilder
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.hadamard import Hadamard
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.sum import Sum
from systemds.scuro.representations.zero_pad import ZeroPad
from tests.scuro.data_generator import ModalityRandomDataGenerator
from tests.scuro.test_multimodal_ga_optimizer import (
    _FakeModality,
    _FakeUnimodalResults,
    _make_task,
)

AGG = AlgebraConfig(explicit_fusion_prep=True)
PAD = AlgebraConfig(explicit_padding="pad")
BOTH = AlgebraConfig(explicit_fusion_prep=True, explicit_padding="pad")
FUSIONS = [Concatenation, Sum, Average, Hadamard]
ALIGNED = {Sum, Average, Hadamard}


def _reps():
    """Modality 0 is a (4, 8) sequence; 1 and 2 are vectors of width 3 and 8."""
    gen = ModalityRandomDataGenerator()
    data, _ = gen.create_2d_modality(6, (4, 8))
    seq = TransformedModality(gen, "test")
    seq.data = data
    reps = {0: seq}
    for mid, width in ((1, 3), (2, 8)):
        reps[mid] = gen.create1DModality(6, width, ModalityType.TIMESERIES)
    for mid, rep in reps.items():
        rep.modality_id = mid
    return {mid: [rep] for mid, rep in reps.items()}


def _expected(algebra, op, leaves):
    out = []
    if algebra.explicit_fusion_prep and 0 in leaves:
        out.append(("AggregatedRepresentation", 0))
    # Padding needs every operand to be a vector first.
    if algebra.explicit_padding == "pad" and op in ALIGNED:
        if algebra.explicit_fusion_prep or 0 not in leaves:
            out.append(("ZeroPad", 1))
    return out


def _prep_nodes(dag):
    return sorted(
        (n.operation.__name__, dag.get_node_by_id(n.inputs[0]).modality_id)
        for n in dag.filter_connected_nodes(dag.nodes)
        if n.operation in (AggregatedRepresentation, ZeroPad)
    )


def _fused(dag, reps):
    out = dag.execute([m for ms in reps.values() for m in ms], enable_cache=False)
    return np.asarray(out[dag.root_node_id] if isinstance(out, dict) else out.data)


def _assert_equivalent(legacy, explicit, reps, expected):
    assert _prep_nodes(legacy) == []
    assert _prep_nodes(explicit) == expected
    assert canonical_id(legacy, ALL_LAWS, False) == canonical_id(
        explicit, ALL_LAWS, False
    )
    np.testing.assert_array_equal(_fused(legacy, reps), _fused(explicit, reps))


def _ga_dag(algebra, reps, genome):
    optimizer = MultimodalDeapOptimizer(
        [_FakeModality(m) for m in reps],
        _FakeUnimodalResults(reps),
        [_make_task()],
        debug=False,
        algebra=algebra,
    )
    optimizer._current_task_name = "task0"
    return optimizer._genome_to_dag(genome)


class TestExplicitPrep(unittest.TestCase):
    def test_ga_genome_to_dag(self):
        cases = itertools.product([AGG, PAD, BOTH], FUSIONS, [(0, 1), (1, 2)])
        for algebra, op, leaves in cases:
            with self.subTest(algebra=algebra, op=op.__name__, leaves=leaves):
                reps = _reps()
                genome = DagGenome([(m, 0) for m in leaves], (0, 1), {"": op})
                _assert_equivalent(
                    _ga_dag(AlgebraConfig(), reps, genome),
                    _ga_dag(algebra, reps, genome),
                    reps,
                    _expected(algebra, op, leaves),
                )

    def test_exhaustive_enumerator(self):
        for algebra in (AGG, BOTH):
            with self.subTest(algebra=algebra):
                reps = _reps()
                runs = []
                for explicit in (AlgebraConfig(), algebra):
                    with tempfile.TemporaryDirectory() as tmp:
                        optimizer = MultimodalOptimizer(
                            [_FakeModality(m) for m in reps],
                            _FakeUnimodalResults(reps),
                            [_make_task()],
                            debug=False,
                            checkpoint_dir=tmp,
                            algebra=AlgebraConfig(
                                exhaustive_dedup_permutations=True,
                                explicit_fusion_prep=explicit.explicit_fusion_prep,
                                explicit_padding=explicit.explicit_padding,
                            ),
                        )
                    optimizer.fusion_operators = FUSIONS
                    runs.append(
                        list(
                            optimizer._generate_fusion_dags(
                                [0, 1], {0: 0, 1: 0}, "task0"
                            )
                        )
                    )
                self.assertEqual(len(runs[0]), 2 * len(FUSIONS))
                self.assertEqual(len(runs[1]), 2 * len(FUSIONS))
                for legacy, explicit, op in zip(*runs, FUSIONS * 2):
                    _assert_equivalent(
                        legacy, explicit, reps, _expected(algebra, op, (0, 1))
                    )
                self.assertEqual(
                    optimizer.algebra_stats["explicit_prep_plans"], 2 * len(FUSIONS)
                )

    def test_counts_and_typechecks_padded_plan(self):
        reps = _reps()
        dag = _ga_dag(BOTH, reps, DagGenome([(0, 0), (1, 0)], (0, 1), {"": Sum}))
        stats = Counter()
        count_explicit_prep(stats, dag)
        self.assertEqual(
            stats,
            {
                "explicit_prep_plans": 1,
                "explicit_padding_plans": 1,
                "explicit_aggregations": 1,
                "explicit_paddings": 1,
                "explicit_padding_ratio_sum": 8 / 3,
                "explicit_padding_ratio_max": 8 / 3,
            },
        )
        leaves = {(m, 0): r[0] for m, r in reps.items()}
        types = {k: r.modality_type for k, r in leaves.items()}
        leaf_stats = {k: sample_stats(r) for k, r in leaves.items()}
        self.assertEqual(typecheck(dag, types, leaf_stats), [])

    def test_zero_pad_is_not_registered(self):
        registry = Registry()
        lists = [
            registry._representations,
            registry._context_operators,
            registry._dimensionality_reduction_operators,
        ]
        for ops in (ops for lst in lists for ops in lst.values()):
            self.assertNotIn(ZeroPad, ops)

    def test_unbound_genome_to_dag_and_bare_checker(self):
        reps = _reps()
        genome = DagGenome([(0, 0), (1, 0)], (0, 1), {"": Sum})
        dag = MultimodalDeapOptimizer._genome_to_dag(None, genome)
        self.assertEqual(_prep_nodes(dag), [])
        checker = object.__new__(MultimodalDeapOptimizer)
        checker.max_feature_dim = None
        checker.k_best_representations = {"task0": reps}
        self.assertIsNone(checker._oversized_reason("task0", genome))

    def test_cse_builder(self):
        for params, wrapped in (({}, True), ({"preserve_leading_axis": True}, False)):
            with self.subTest(params=params):
                leaf = SimpleNamespace(
                    modality_id="t", stats=RepresentationStats(5, (4, 8))
                )
                builder = CSEAwareDAGBuilder(BOTH, leaves=[leaf])
                leaf_id = builder.create_leaf_node("t")
                root = builder.create_operation_node(
                    Concatenation, [leaf_id, leaf_id], params
                )
                builder.build(root)
                self.assertEqual(
                    builder.get_node(root).inputs != [leaf_id, leaf_id], wrapped
                )
                self.assertEqual(
                    builder.algebra_stats["explicit_aggregations"], 2 * wrapped
                )


if __name__ == "__main__":
    unittest.main()
