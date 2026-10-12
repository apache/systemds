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

import numpy as np
import pytest

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


@pytest.mark.parametrize("case", OPERANDS)
@pytest.mark.parametrize("op", OPERATORS, ids=lambda op: op.__name__)
def test_predicted_stats_match_execution(op, case):
    rng = np.random.default_rng(0)
    operands = [
        rng.standard_normal(shape).astype(np.float32) for shape in OPERANDS[case]
    ]
    stats = [RepresentationStats(len(x), x.shape[1:]) for x in operands]
    fused = _fuse(op, {}, *operands)
    assert tuple(op().get_output_stats(stats).output_shape) == fused.shape[1:]
    assert op().estimate_peak_memory_bytes(stats)["cpu_peak_bytes"] > fused.nbytes
