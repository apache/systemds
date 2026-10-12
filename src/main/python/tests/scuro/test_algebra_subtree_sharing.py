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
from unittest.mock import patch

import numpy as np
import pytest

from systemds.scuro.drsearch import multimodal_ga_optimizer as ga
from systemds.scuro.drsearch.algebra import AlgebraConfig
from systemds.scuro.drsearch.representation_dag import RepresentationDAGBuilder
from systemds.scuro.drsearch.trial_result_cache import TrialResultCache
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.lstm import LSTM
from tests.scuro.data_generator import TestSVM, TestTask
from tests.scuro.test_multimodal_ga_optimizer import (
    _build_real_optimizer,
    _make_real_representation,
)

LSTM_PARAMS = {"width": 8, "epochs": 2}
SPECS = [("accuracy", "max")]


class _RecordingSVM(TestSVM):
    inputs = []

    def fit(self, X, y, X_test, y_test):
        self.inputs.append(np.array(X))
        return super().fit(X, y, X_test, y_test)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(ga, "_SUBTREE_CACHE", TrialResultCache(2**30, 0.0))
    task = TestTask("share_task", "share_model", 40)
    task.model = _RecordingSVM("share_model")
    modalities = [_make_real_representation(i, 40, 4) for i in range(3)]
    _RecordingSVM.inputs = []
    return task, modalities


def _dag(leaves, root_op=None):
    b = RepresentationDAGBuilder()
    ids = [b.create_leaf_node(i, 0) for i in leaves]
    node = b.create_operation_node(LSTM, ids[:2], dict(LSTM_PARAMS))
    if root_op is not None:
        node = b.create_operation_node(root_op, [node, ids[2]], {})
    return b.build(node)


def _evaluate(task, modalities, dag, share):
    _RecordingSVM.inputs = []
    _, payload = ga._evaluate_genome_body(dag, task, modalities, SPECS, share)
    return payload, _RecordingSVM.inputs


def test_fitted_subtree_is_reused_bit_identically(setup):
    task, modalities = setup
    _evaluate(task, modalities, _dag([0, 1]), True)
    shared, shared_inputs = _evaluate(
        task, modalities, _dag([0, 1, 2], Concatenation), True
    )
    _, fresh_inputs = _evaluate(task, modalities, _dag([0, 1, 2], Concatenation), False)

    assert shared["subtree_cache"]["hits"] == len(task.evaluation_splits)
    assert len(shared_inputs) == len(fresh_inputs) == len(task.evaluation_splits)
    assert all(np.array_equal(a, b) for a, b in zip(shared_inputs, fresh_inputs))


def test_operand_order_is_part_of_the_key(setup):
    task, modalities = setup
    _evaluate(task, modalities, _dag([0, 1]), True)
    swapped, _ = _evaluate(task, modalities, _dag([1, 0]), True)
    assert swapped["subtree_cache"]["hits"] == 0


@pytest.mark.parametrize("share", [False, True])
def test_switch_controls_call_and_stats(share):
    optimizer, task = _build_real_optimizer(
        fusion_ops=[LSTM], algebra=AlgebraConfig(share_fused_subtrees=share)
    )
    genome = optimizer._random_genome(task.model.name)
    payload = {
        "train_score": {},
        "val_score": {"accuracy": 0.5},
        "test_score": {},
        "runtime": 1.0,
        "task_time": 0.5,
        "representation_time": 0.5,
        "subtree_cache": {"hits": 3, "misses": 2, "saved_seconds": 4.0, "bytes": 10},
    }
    with patch.object(
        ga, "_evaluate_genome_body", return_value=((0.5,), payload)
    ) as body:
        optimizer._evaluate_genome(genome, task)
    assert len(body.call_args.args) == 4 + share
    assert optimizer.algebra_stats["subtree_cache_hits"] == 3
