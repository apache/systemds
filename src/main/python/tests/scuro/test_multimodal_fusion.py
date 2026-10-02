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

import numpy as np

from systemds.scuro.drsearch.multimodal_optimizer import (
    MultimodalOptimizer,
    OptimizationResult,
)
from systemds.scuro.drsearch.unimodal_optimizer import UnimodalOptimizer
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.representations.average import Average
from systemds.scuro.drsearch.operator_registry import Registry

from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.timeseries_representations import Min, Max
from tests.scuro.data_generator import (
    TestDataLoader,
    ModalityRandomDataGenerator,
    TestTask,
)

from systemds.scuro.modality.type import ModalityType
from unittest.mock import patch


class TestMultimodalRepresentationOptimizer(unittest.TestCase):
    test_file_path = None
    data_generator = None
    num_instances = 0

    # @classmethod
    # def setUpClass(cls):
    #     cls.num_instances = 10
    #     cls.mods = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]
    #     cls.indices = np.array(range(cls.num_instances))

    # Note: Multimodal fusion is being refactored and not yet ready for testing
    # def test_multimodal_fusion(self):
    #     task = TestTask("MM_Fusion_Task1", "Test1", self.num_instances)

    #     audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
    #         self.num_instances, 1000
    #     )
    #     text_data, text_md = ModalityRandomDataGenerator().create_text_data(
    #         self.num_instances
    #     )

    #     audio = UnimodalModality(
    #         TestDataLoader(
    #             self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
    #         )
    #     )
    #     text = UnimodalModality(
    #         TestDataLoader(
    #             self.indices, None, ModalityType.TEXT, text_data, str, text_md
    #         )
    #     )

    #     with patch.object(
    #         Registry,
    #         "_representations",
    #         {
    #             ModalityType.TEXT: [W2V],
    #             ModalityType.AUDIO: [Spectrogram],
    #             ModalityType.TIMESERIES: [ResNet],
    #             ModalityType.VIDEO: [ResNet],
    #             ModalityType.EMBEDDING: [],
    #         },
    #     ):
    #         registry = Registry()
    #         registry._fusion_operators = [Average, Concatenation, LSTM]
    #         unimodal_optimizer = UnimodalOptimizer([audio, text], [task], debug=False)
    #         unimodal_optimizer.optimize()
    #         unimodal_optimizer.operator_performance.get_k_best_results(
    #             audio, 2, task, "accuracy"
    #         )
    #         m_o = MultimodalOptimizer(
    #             [audio, text],
    #             unimodal_optimizer.operator_performance,
    #             [task],
    #             debug=False,
    #             min_modalities=2,
    #             max_modalities=3,
    #         )
    #         fusion_results = m_o.optimize(20)

    #         best_results = sorted(
    #             fusion_results[task.model.name],
    #             key=lambda x: getattr(x, "val_score")["accuracy"],
    #             reverse=True,
    #         )[:2]

    #         assert (
    #             best_results[0].val_score["accuracy"]
    #             >= best_results[1].val_score["accuracy"]
    #         )

    # def test_parallel_multimodal_fusion(self):
    #     task = TestTask("MM_Fusion_Task1", "Test2", self.num_instances)
    #
    #     audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
    #         self.num_instances, 1000
    #     )
    #     text_data, text_md = ModalityRandomDataGenerator().create_text_data(
    #         self.num_instances
    #     )
    #
    #     audio = UnimodalModality(
    #         TestDataLoader(
    #             self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
    #         )
    #     )
    #     text = UnimodalModality(
    #         TestDataLoader(
    #             self.indices, None, ModalityType.TEXT, text_data, str, text_md
    #         )
    #     )
    #
    #     with patch.object(
    #         Registry,
    #         "_representations",
    #         {
    #             ModalityType.TEXT: [W2V],
    #             ModalityType.AUDIO: [Spectrogram],
    #             ModalityType.TIMESERIES: [Max, Min],
    #             ModalityType.VIDEO: [ResNet],
    #             ModalityType.EMBEDDING: [],
    #         },
    #     ):
    #         registry = Registry()
    #         registry._fusion_operators = [Average, Concatenation, LSTM]
    #         unimodal_optimizer = UnimodalOptimizer([audio, text], [task], debug=False)
    #         unimodal_optimizer.optimize()
    #         unimodal_optimizer.operator_performance.get_k_best_results(
    #             audio, 2, task, "accuracy"
    #         )
    #         m_o = MultimodalOptimizer(
    #             [audio, text],
    #             unimodal_optimizer.operator_performance,
    #             [task],
    #             debug=False,
    #             min_modalities=2,
    #             max_modalities=3,
    #         )
    #         fusion_results = m_o.optimize(max_combinations=16)
    #         parallel_fusion_results = m_o.optimize_parallel(16, max_workers=2, batch_size=4)
    #
    #         best_results = sorted(
    #             fusion_results[task.model.name],
    #             key=lambda x: getattr(x, "val_score")["accuracy"],
    #             reverse=True,
    #         )
    #
    #         best_results_parallel = sorted(
    #             parallel_fusion_results[task.model.name],
    #             key=lambda x: getattr(x, "val_score")["accuracy"],
    #             reverse=True,
    #         )
    #
    #         # assert len(best_results) == len(best_results_parallel)
    #         for i in range(len(best_results)):
    #             assert (
    #                 best_results[i].val_score["accuracy"]
    #                 == best_results_parallel[i].val_score["accuracy"]
    #             )


class _FakeModality:
    def __init__(self, modality_id):
        self.modality_id = modality_id


class _FakeUnimodalResults:
    """Stand-in for UnimodalOptimizer.operator_performance: hands back a
    fixed list of representations per modality without running any real
    unimodal search."""

    def __init__(self, reps_per_modality):
        self.reps_per_modality = reps_per_modality

    def get_k_best_results(self, modality, task, performance_metric_name):
        reps = self.reps_per_modality.get(modality.modality_id, [])
        return list(range(len(reps))), reps


def _make_task(name="task0"):
    return SimpleNamespace(model=SimpleNamespace(name=name))


def _make_optimizer(reps_per_modality, **kwargs):
    modalities = [_FakeModality(modality_id) for modality_id in reps_per_modality]
    task = _make_task()
    kwargs.setdefault("debug", False)
    # a resumed run loads any checkpoint left in the working directory
    kwargs.setdefault("resume", False)
    optimizer = MultimodalOptimizer(
        modalities, _FakeUnimodalResults(reps_per_modality), [task], **kwargs
    )
    return optimizer, task


class TestFusionDagGeneration(unittest.TestCase):
    def test_modality_combinations_respect_min_max_modalities(self):
        reps = {"m0": [object()], "m1": [object()], "m2": [object()]}
        pairs = [["m0", "m1"], ["m0", "m2"], ["m1", "m2"]]
        for max_modalities, expected in (
            (2, pairs),
            (3, pairs + [["m0", "m1", "m2"]]),
            (5, pairs + [["m0", "m1", "m2"]]),
        ):
            with self.subTest(max_modalities=max_modalities):
                optimizer, _ = _make_optimizer(
                    reps, min_modalities=2, max_modalities=max_modalities
                )
                self.assertEqual(
                    list(optimizer._generate_modality_combinations()), expected
                )

    def test_representation_combinations_pick_one_per_modality(self):
        reps = {"m0": [object(), object()], "m1": [object(), object(), object()]}
        optimizer, task = _make_optimizer(reps)
        combinations = list(
            optimizer._generate_representation_combinations(
                ["m0", "m1"], task.model.name
            )
        )
        self.assertEqual(
            combinations,
            [
                {"m0": 0, "m1": 0},
                {"m0": 0, "m1": 1},
                {"m0": 0, "m1": 2},
                {"m0": 1, "m1": 0},
                {"m0": 1, "m1": 1},
                {"m0": 1, "m1": 2},
            ],
        )

    def test_fusion_dags_contain_every_selected_representation(self):
        reps = {
            "m0": [object(), object()],
            "m1": [object()],
            "m2": [object(), object()],
        }
        optimizer, _ = _make_optimizer(reps)
        optimizer.fusion_operators = [Concatenation, Average]
        selection = {"m0": 1, "m1": 0, "m2": 1}
        dags = list(optimizer._generate_fusion_dags(["m0", "m1", "m2"], selection))
        # the same tree can be yielded more than once
        self.assertTrue(dags)
        for dag in dags:
            leaves = [node for node in dag.nodes if not node.inputs]
            fusions = [node for node in dag.nodes if node.inputs]
            self.assertEqual(
                {(leaf.modality_id, leaf.representation_index) for leaf in leaves},
                set(selection.items()),
            )
            self.assertEqual(len(fusions), len(leaves) - 1)
            self.assertTrue(all(len(node.inputs) == 2 for node in fusions))
            self.assertIn(dag.root_node_id, {node.node_id for node in fusions})

    def test_fusion_dags_use_every_fusion_operator(self):
        reps = {"m0": [object()], "m1": [object()]}
        optimizer, _ = _make_optimizer(reps)
        optimizer.fusion_operators = [Concatenation, Average]
        used = {
            node.operation
            for dag in optimizer._generate_fusion_dags(["m0", "m1"], {"m0": 0, "m1": 0})
            for node in dag.nodes
            if node.inputs
        }
        self.assertEqual(used, {Concatenation, Average})


class TestConstructorValidation(unittest.TestCase):
    def test_min_modalities_clamped_to_two(self):
        optimizer, _ = _make_optimizer(
            {"m0": [object()], "m1": [object()]}, min_modalities=1
        )
        self.assertEqual(optimizer.min_modalities, 2)

    def test_max_modalities_defaults_to_number_of_modalities(self):
        optimizer, _ = _make_optimizer(
            {"m0": [object()], "m1": [object()], "m2": [object()]}
        )
        self.assertEqual(optimizer.max_modalities, 3)

    def test_k_best_representations_hold_the_cached_data(self):
        reps = {"m0": [object(), object()], "m1": [object()]}
        optimizer, task = _make_optimizer(reps)
        k_best = optimizer.k_best_representations[task.model.name]
        self.assertEqual(set(k_best), {"m0", "m1"})
        for modality_id in reps:
            self.assertIs(k_best[modality_id], reps[modality_id])


def _stored_result(dag, task):
    return OptimizationResult(dag=dag, task_name=task.model.name)


class TestOptimizeLoop(unittest.TestCase):
    def test_stops_at_max_combinations(self):
        optimizer, task = _make_optimizer(
            {"m0": [object()], "m1": [object()], "m2": [object()]}
        )
        optimizer.fusion_operators = [Concatenation]
        with patch.object(
            optimizer, "_evaluate_dag", side_effect=_stored_result
        ) as evaluate_dag:
            results = optimizer.optimize(max_combinations=5)

        self.assertEqual(evaluate_dag.call_count, 5)
        self.assertEqual(len(results[task.model.name]), 5)

    def test_drops_failed_evaluations(self):
        optimizer, task = _make_optimizer(
            {"m0": [object()], "m1": [object()], "m2": [object()]}
        )
        optimizer.fusion_operators = [Concatenation]
        call_counter = {"n": 0}

        def flaky_evaluate(dag, _task):
            call_counter["n"] += 1
            if call_counter["n"] % 2 == 0:
                return None
            return _stored_result(dag, _task)

        with patch.object(optimizer, "_evaluate_dag", side_effect=flaky_evaluate):
            results = optimizer.optimize(max_combinations=6)

        self.assertEqual(call_counter["n"], 6)
        self.assertEqual(len(results[task.model.name]), 3)

    def test_keeps_results_and_budget_per_task(self):
        tasks = [_make_task("task_a"), _make_task("task_b")]
        optimizer = MultimodalOptimizer(
            [_FakeModality("m0"), _FakeModality("m1")],
            _FakeUnimodalResults({"m0": [object()], "m1": [object()]}),
            tasks,
            debug=False,
            resume=False,
        )
        optimizer.fusion_operators = [Concatenation]
        with patch.object(optimizer, "_evaluate_dag", side_effect=_stored_result):
            results = optimizer.optimize(max_combinations=2)

        self.assertEqual(
            {name: len(stored) for name, stored in results.items()},
            {"task_a": 2, "task_b": 2},
        )
        for name, stored in results.items():
            self.assertEqual({result.task_name for result in stored}, {name})


if __name__ == "__main__":
    unittest.main()
