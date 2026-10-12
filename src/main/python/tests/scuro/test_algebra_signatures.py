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

import pytest

from systemds.scuro.drsearch.algebra import ANY_TYPE, operator_signature
from systemds.scuro.drsearch.operator_registry import Registry, load_builtin_operators
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.representations.mlp_averaging import MLPAveraging
from systemds.scuro.representations.sum import Sum
from systemds.scuro.representations.window_aggregation import WindowAggregation

load_builtin_operators(ANY_TYPE, include_fusion=True)
REGISTERED = sorted(
    {
        (op, kind, modality)
        for kind, lists in [
            ("encode", Registry._representations),
            ("context", Registry._context_operators),
            ("reduce", Registry._dimensionality_reduction_operators),
        ]
        for modality, ops in lists.items()
        for op in ops
    },
    key=lambda t: (t[0].__name__, t[1], t[2].name),
) + [(op, "fuse", None) for op in Registry().get_fusion_operators()]


@pytest.mark.parametrize(
    "op, kind, modality",
    REGISTERED,
    ids=[
        f"{op.__name__}-{kind}-{getattr(m, 'name', '')}" for op, kind, m in REGISTERED
    ],
)
def test_every_registered_operator_has_a_signature(op, kind, modality):
    sig = operator_signature(op)
    assert sig.kind == kind
    assert modality is None or modality in sig.input_types
    if kind in ("encode", "fuse"):
        assert isinstance(sig.output_type, ModalityType)


@pytest.mark.parametrize(
    "op, params, expected",
    [
        (Sum, {}, ("fuse", "n", "vector", "vector")),
        (
            Concatenation,
            {"preserve_leading_axis": False},
            ("fuse", "n", "vector", "vector"),
        ),
        (Concatenation, {"preserve_leading_axis": True}, ("fuse", "n", "seq", "seq")),
        (LSTM, {}, ("fuse", "n", "seq", "vector")),
        (AggregatedRepresentation, {}, ("aggregate", 1, "seq", "vector")),
        (WindowAggregation, {}, ("context", 1, "seq", "windows")),
        (MLPAveraging, {}, ("reduce", 1, "vector", "vector")),
        (None, {}, ("task", 1, "any", "any")),
    ],
)
def test_rank_rules(op, params, expected):
    sig = operator_signature(op, params)
    assert (sig.kind, sig.arity, sig.input_rank, sig.output_rank) == expected
