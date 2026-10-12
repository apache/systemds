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
from unittest.mock import patch

from systemds.scuro.drsearch.algebra import AlgebraConfig
from systemds.scuro.drsearch.multimodal_ga_optimizer import DagGenome
from systemds.scuro.representations.sum import Sum
from tests.scuro.test_multimodal_ga_optimizer import MODULE, _make_optimizer


def _body(_dag, _task, _modalities, _objective_specs):
    return (0.5,), {
        "train_score": {},
        "val_score": {"accuracy": 0.5},
        "test_score": {},
        "runtime": 2.0,
        "task_time": 1.0,
        "representation_time": 1.0,
    }


class TestCanonicalGADedup(unittest.TestCase):
    def test_commuted_sum_evaluated_once_when_on(self):
        leaves = [("m0", 0), ("m1", 0)]
        genomes = [DagGenome(leaves, tree, {"": Sum}) for tree in ((0, 1), (1, 0))]
        for algebra, evaluations in (
            (AlgebraConfig(), 2),
            (AlgebraConfig(laws=frozenset({"comm"}), ga_canonical_dedup=True), 1),
        ):
            with self.subTest(algebra=algebra):
                optimizer, _, task = _make_optimizer(
                    n_modalities=2, max_workers=1, algebra=algebra
                )
                optimizer._current_task_name = task.model.name
                population = [optimizer._make_individual(g) for g in genomes]
                with patch(f"{MODULE}._evaluate_genome_body", side_effect=_body):
                    optimizer._evaluate_population(population, task)
                results = optimizer.optimization_results["task0"]
                self.assertEqual(len(results), evaluations)
                avoided = 2 - evaluations
                stats = optimizer.algebra_stats
                self.assertEqual(stats["ga_evaluations_avoided"], avoided)
                self.assertEqual(stats["ga_seconds_avoided"], 2.0 * avoided)
                self.assertEqual(stats["ga_unique_legacy"], 2 * avoided)
                self.assertEqual(stats["ga_unique_canonical"], avoided)
