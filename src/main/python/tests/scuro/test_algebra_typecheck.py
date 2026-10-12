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

from types import SimpleNamespace

import numpy as np
import pytest

from systemds.scuro.drsearch.algebra import AlgebraConfig, typecheck
from systemds.scuro.drsearch.multimodal_ga_optimizer import (
    DagGenome,
    MultimodalDeapOptimizer,
)
from systemds.scuro.drsearch.representation_dag import (
    CSEAwareDAGBuilder,
    RepresentationDAGBuilder,
)
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.mfcc import MFCC
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.sum import Sum
from tests.scuro.test_multimodal_ga_optimizer import _FakeUnimodalResults, _make_task

TEXT, EMB = ModalityType.TEXT, ModalityType.EMBEDDING
SEQ, VEC = RepresentationStats(5, (4, 8)), RepresentationStats(5, (3,))


def _check(op, params, leaves, with_stats=True):
    builder = RepresentationDAGBuilder()
    ids = [builder.create_leaf_node(f"m{i}", 0) for i in range(len(leaves))]
    dag = builder.build(builder.create_operation_node(op, ids, params))
    types = {(f"m{i}", 0): t for i, (t, _) in enumerate(leaves)}
    stats = {(f"m{i}", 0): s for i, (_, s) in enumerate(leaves)}
    violations = typecheck(dag, types, stats if with_stats else None)
    return sorted((v.severity, v.kind) for v in violations)


@pytest.mark.parametrize(
    "op, params, leaves, expected",
    [
        (BoW, {}, [(TEXT, None)], []),
        (MFCC, {}, [(TEXT, None)], [("error", "input_type")]),
        (Sum, {}, [(EMB, VEC)], [("error", "arity")]),
        (BoW, {}, [(TEXT, None), (TEXT, None)], [("error", "arity")]),
        (Sum, {}, [(EMB, VEC), (EMB, VEC)], []),
        (
            Sum,
            {},
            [(EMB, SEQ), (EMB, VEC)],
            [("implicit", "hidden_aggregation"), ("implicit", "zero_padding")],
        ),
        (
            Concatenation,
            {},
            [(EMB, SEQ), (EMB, VEC)],
            [("implicit", "hidden_aggregation")],
        ),
        (Concatenation, {"preserve_leading_axis": True}, [(EMB, SEQ), (EMB, SEQ)], []),
    ],
)
def test_violations(op, params, leaves, expected):
    assert _check(op, params, leaves) == expected


def test_shapes_unknown_without_leaf_stats():
    assert _check(Sum, {}, [(EMB, SEQ), (EMB, VEC)], with_stats=False) == []


@pytest.mark.parametrize(
    "mode, raises, counted",
    [("off", False, 0), ("warn", False, 1), ("strict", True, 1)],
)
def test_builder_modes(mode, raises, counted):
    modality = SimpleNamespace(modality_id="t", modality_type=TEXT, stats=None)
    builder = CSEAwareDAGBuilder(AlgebraConfig(typecheck=mode), leaves=[modality])
    root = builder.create_operation_node(MFCC, [builder.create_leaf_node("t")], {})
    if raises:
        with pytest.raises(ValueError, match="input_type"):
            builder.build(root)
    else:
        builder.build(root)
    assert builder.algebra_stats["typecheck_input_type"] == counted


def test_ga_counts_hidden_operators():
    reps = {
        "m0": [SimpleNamespace(modality_type=EMB, data=np.zeros((5, 4, 8)))],
        "m1": [SimpleNamespace(modality_type=EMB, data=np.zeros((5, 3)))],
    }
    optimizer = MultimodalDeapOptimizer(
        [SimpleNamespace(modality_id=m) for m in reps],
        _FakeUnimodalResults(reps),
        [_make_task()],
        debug=False,
        algebra=AlgebraConfig(typecheck="strict"),
    )
    genome = DagGenome([("m0", 0), ("m1", 0)], (0, 1), {"": Sum})
    assert optimizer._oversized_reason("task0", genome) is None
    stats = optimizer.algebra_stats
    assert stats["typecheck_hidden_aggregation"] == 1
    assert stats["typecheck_zero_padding:Sum"] == 1
