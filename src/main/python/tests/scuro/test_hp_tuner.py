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

import sys
import unittest
from types import SimpleNamespace

import numpy as np

from systemds.scuro import Mean
from systemds.scuro.drsearch.multimodal_optimizer import MultimodalOptimizer
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.color_histogram import ColorHistogram
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.unimodal_optimizer import UnimodalOptimizer

from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.covarep_audio_features import (
    ZeroCrossing,
    Spectral,
    Pitch,
)
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.bow import BoW
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.resnet import ResNet
from tests.scuro.data_generator import (
    ModalityRandomDataGenerator,
    TestDataLoader,
    TestTask,
)

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.hyperparameter_tuner import (
    HyperparamResult,
    HyperparamResults,
    HyperparameterTuner,
    _apply_pushdown_trial_params,
    _apply_trial_params_to_node,
    _expand_aggregation_param_specs,
    _is_aggregated_representation_operation,
    _is_window_operation,
    _materialize_node_params,
    _param_values_to_spec,
    _window_input_stats,
)
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDAGBuilder,
    RepresentationDag,
    RepresentationNode,
)
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.window_aggregation import WindowAggregation

from unittest.mock import patch


class TestHPTuner(unittest.TestCase):
    data_generator = None
    num_instances = 0

    @classmethod
    def setUpClass(cls):
        cls.num_instances = 10
        cls.mods = [
            ModalityType.VIDEO,
            ModalityType.AUDIO,
            ModalityType.TEXT,
            ModalityType.IMAGE,
        ]
        cls.indices = np.array(range(cls.num_instances))
        cls.tasks = [
            TestTask("UnimodalRepresentationTask1", "TestSVM1", cls.num_instances),
            TestTask("UnimodalRepresentationTask2", "TestSVM2", cls.num_instances),
        ]

    def test_hp_tuner_for_text_modality(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        self.run_hp_for_modality([text])

    # TODO: Add once the final multimodal optimizer is implemented
    # def test_multimodal_hp_tuning(self):
    #     audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
    #         self.num_instances, 3000
    #     )
    #     audio = UnimodalModality(
    #         TestDataLoader(
    #             self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
    #         )
    #     )

    #     text_data, text_md = ModalityRandomDataGenerator().create_text_data(
    #         self.num_instances
    #     )
    #     text = UnimodalModality(
    #         TestDataLoader(
    #             self.indices, None, ModalityType.TEXT, text_data, str, text_md
    #         )
    #     )

    #     self.run_hp_for_modality(
    #         [audio, text], multimodal=True, tune_unimodal_representations=False
    #     )

    def test_hp_tuner_for_image_modality(self):
        image_data, image_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 1
        )
        image = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.IMAGE, image_data, np.float32, image_md
            )
        )
        self.run_hp_for_modality([image])

    def run_hp_for_modality(
        self, modalities, multimodal=False, tune_unimodal_representations=False
    ):
        with patch.object(
            Registry,
            "_representations",
            {
                ModalityType.TEXT: [BoW, W2V],
                ModalityType.AUDIO: [Spectrogram, ZeroCrossing, Spectral, Pitch],
                ModalityType.TIMESERIES: [Mean],
                ModalityType.VIDEO: [ResNet],
                ModalityType.IMAGE: [ResNet, ColorHistogram],
                ModalityType.EMBEDDING: [],
            },
        ), patch.object(Registry, "_fusion_operators", [LSTM]):
            unimodal_optimizer = UnimodalOptimizer(
                modalities, self.tasks, False, k=2, max_num_workers=1
            )
            unimodal_optimizer.optimize()

            hp = HyperparameterTuner(
                modalities,
                self.tasks,
                unimodal_optimizer.operator_performance,
                n_jobs=1,
            )

            if multimodal:
                m_o = MultimodalOptimizer(
                    modalities,
                    unimodal_optimizer.operator_performance,
                    self.tasks,
                    debug=False,
                    min_modalities=2,
                    max_modalities=3,
                )
                fusion_results = m_o.optimize(20)

                hp.tune_multimodal_representations(
                    fusion_results,
                    k=1,
                    optimize_unimodal=tune_unimodal_representations,
                    max_eval_per_rep=10,
                )

            else:
                hp.tune_unimodal_representations(max_eval_per_rep=2)

            assert len(hp.optimization_results.results) == len(self.tasks)
            if multimodal:
                if tune_unimodal_representations:
                    assert (
                        len(
                            hp.optimization_results.results[self.tasks[0].model.name][0]
                        )
                        == 1
                    )
                else:
                    assert (
                        len(
                            hp.optimization_results.results[self.tasks[0].model.name][
                                "mm_results"
                            ]
                        )
                        == 1
                    )
            else:
                assert (
                    len(hp.optimization_results.results[self.tasks[0].model.name]) == 1
                )
                modality_id = modalities[0].modality_id
                assert (
                    len(
                        hp.optimization_results.results[self.tasks[0].model.name][
                            modality_id
                        ]
                    )
                    == 2
                )

    def test_evaluate_configs_deduplicates_candidates(self):
        class DummyOptimizationResults:
            def get_k_best_results(self, modality, task, performance_metric_name):
                return [], []

        task = SimpleNamespace(model=SimpleNamespace(name="dummy_task"))
        hp = HyperparameterTuner(
            modalities=[],
            tasks=[task],
            optimization_results=DummyOptimizationResults(),
            n_jobs=1,
        )

        eval_calls = {"count": 0}

        def fake_evaluate(
            dag, params, node_order, modality_ids, task, modalities_override=None
        ):
            eval_calls["count"] += 1
            return params, float(params["x"])

        hp.evaluate_dag_config = fake_evaluate

        candidate_configs = [
            {"x": 1},
            {"x": 1},
            {"x": 2},
            {"x": 2},
            {"x": 1},
        ]
        seen_configs = {}

        results = hp._evaluate_configs(
            dag=None,
            task=None,
            node_order=[],
            modality_ids=[],
            modalities_override=None,
            candidate_configs=candidate_configs,
            seen_configs=seen_configs,
        )

        self.assertEqual(eval_calls["count"], 2)
        self.assertEqual(len(seen_configs), 2)
        self.assertEqual([r[0]["x"] for r in results], [1, 2])


class _NarrowingAggregation:
    """An aggregation that drops window sizes larger than the input. It stands
    in for operators that adapt their domain to the data they receive."""

    def __init__(self):
        self.parameters = {"window_size": [2, 4, 8]}

    def filter_parameter_domain(self, name, values, input_stats):
        return [value for value in values if value <= input_stats.output_shape[0]]


class TestSearchSpaceConstruction(unittest.TestCase):
    """Every operator declares its parameters in its own shape. These helpers
    translate them into the specifications optuna samples from."""

    def test_param_values_to_spec_maps_a_list_to_a_categorical_domain(self):
        spec = _param_values_to_spec("node0-window_size", [2, 4, 8])
        self.assertEqual(
            spec,
            {"name": "node0-window_size", "type": "categorical", "domain": [2, 4, 8]},
        )

    def test_param_values_to_spec_maps_an_integer_pair_to_an_integer_domain(self):
        # a pair describes a range and is sorted, so the declared order does
        # not matter
        spec = _param_values_to_spec("node0-n_mfcc", (20, 13))
        self.assertEqual(spec["type"], "integer")
        self.assertEqual(spec["domain"], (13, 20))

    def test_param_values_to_spec_maps_a_float_pair_to_a_real_domain(self):
        spec = _param_values_to_spec("node0-ratio", (0.9, 0.1))
        self.assertEqual(spec["type"], "real")
        self.assertEqual(spec["domain"], (0.1, 0.9))

    def test_param_values_to_spec_wraps_a_single_value_in_a_categorical_domain(self):
        # a fixed value becomes a domain too, so every parameter reaches optuna
        # in the same shape
        spec = _param_values_to_spec("node0-aggregation", "mean")
        self.assertEqual(spec["type"], "categorical")
        self.assertEqual(spec["domain"], ["mean"])

    def test_param_values_to_spec_accepts_any_other_iterable(self):
        spec = _param_values_to_spec("node0-window_size", range(3))
        self.assertEqual(spec["type"], "categorical")
        self.assertEqual(spec["domain"], [0, 1, 2])

    def test_param_values_to_spec_returns_none_for_values_without_a_domain(self):
        for param_values in (None, {"window_size": 4}, range(0)):
            with self.subTest(param_values=param_values):
                self.assertIsNone(_param_values_to_spec("node0-p", param_values))

    def test_param_values_to_spec_returns_none_when_the_value_cannot_be_listed(self):
        # a zero dimensional array offers __iter__ but raises on list()
        self.assertIsNone(_param_values_to_spec("node0-p", np.array(5)))

    def test_window_input_stats_reports_the_window_size_as_the_input_shape(self):
        stats = _window_input_stats({"window_size": 8})
        self.assertEqual(stats.num_instances, 1)
        self.assertEqual(stats.output_shape, (8,))

    def test_window_input_stats_returns_none_without_a_window_size(self):
        for node_parameters in (None, {}, {"aggregation_function": Aggregation}):
            with self.subTest(node_parameters=node_parameters):
                self.assertIsNone(_window_input_stats(node_parameters))

    def test_expand_aggregation_param_specs_expands_the_nested_parameters(self):
        specs = _expand_aggregation_param_specs("node0", Aggregation)
        # Aggregation offers two nested names, but pad_modality carries no
        # domain and is skipped
        self.assertEqual(len(specs), 1)
        self.assertEqual(
            specs[0]["name"], "node0-aggregation_function_aggregation_function"
        )
        self.assertIn("mean", specs[0]["domain"])

    def test_expand_aggregation_param_specs_narrows_the_domain_to_the_input(self):
        specs = _expand_aggregation_param_specs(
            "node0", _NarrowingAggregation, _window_input_stats({"window_size": 4})
        )
        self.assertEqual([spec["domain"] for spec in specs], [[2, 4]])

    def test_expand_aggregation_param_specs_returns_nothing_for_an_instance(self):
        self.assertEqual(_expand_aggregation_param_specs("node0", Aggregation()), [])

    def test_expand_aggregation_param_specs_returns_nothing_without_nested_names(self):
        class _PlainAggregation:
            pass

        self.assertEqual(
            _expand_aggregation_param_specs("node0", _PlainAggregation), []
        )

    def test_expand_aggregation_param_specs_returns_nothing_for_a_failing_class(self):
        # nested names are looked up by class name, so a class called
        # Aggregation reports them before any instance exists -- only then can
        # building the instance still fail
        class Aggregation:
            def __init__(self):
                raise ValueError("cannot be built")

        self.assertEqual(_expand_aggregation_param_specs("node0", Aggregation), [])


class TestApplyTrialParams(unittest.TestCase):
    """A trial hands back one flat value per parameter. Putting a value back is
    not always a top level assignment: a pushed down aggregation keeps its
    parameters in a nested key."""

    def _pushdown_node_parameters(self):
        # pushdown_aggregation moves the parameters of an aggregation node into
        # this nested key and leaves the representation parameters on top
        return {"layer": "avgpool", "_pushdown_aggregation": {"aggregation": "mean"}}

    def test_apply_pushdown_trial_params_moves_nested_values_into_the_pushdown(self):
        result = _apply_pushdown_trial_params(
            self._pushdown_node_parameters(),
            {"aggregation_function_pad_modality": False},
        )
        self.assertEqual(
            result["_pushdown_aggregation"],
            {"aggregation": "mean", "aggregation_function_pad_modality": False},
        )

    def test_apply_pushdown_trial_params_renames_the_aggregation_value(self):
        # AggregatedRepresentation reads aggregation_function_aggregation_function
        # before the plain aggregation key, so the sampled value wins
        result = _apply_pushdown_trial_params(
            self._pushdown_node_parameters(), {"aggregation": "max"}
        )
        self.assertEqual(
            result["_pushdown_aggregation"][
                "aggregation_function_aggregation_function"
            ],
            "max",
        )

    def test_apply_pushdown_trial_params_keeps_other_values_at_the_top_level(self):
        result = _apply_pushdown_trial_params(
            self._pushdown_node_parameters(),
            {"layer": "fc", "_pushdown_aggregation": "ignored"},
        )
        self.assertEqual(result["layer"], "fc")
        self.assertEqual(result["_pushdown_aggregation"], {"aggregation": "mean"})

    def test_apply_pushdown_trial_params_leaves_the_base_parameters_unchanged(self):
        base_params = self._pushdown_node_parameters()
        _apply_pushdown_trial_params(base_params, {"aggregation": "max", "layer": "fc"})
        self.assertEqual(base_params, self._pushdown_node_parameters())

    def test_apply_pushdown_trial_params_removes_nested_keys_from_the_top_level(self):
        # a nested key left on top would reach the representation instead of
        # the aggregation
        base_params = {
            "aggregation_function_pad_modality": True,
            "_pushdown_aggregation": {},
        }
        result = _apply_pushdown_trial_params(base_params, {"n_components": 4})
        self.assertNotIn("aggregation_function_pad_modality", result)
        self.assertEqual(result["n_components"], 4)

    def test_apply_trial_params_to_node_routes_a_pushdown_node_to_the_merge(self):
        node = RepresentationNode(
            node_id="n0",
            operation=WindowAggregation,
            inputs=[],
            parameters=self._pushdown_node_parameters(),
        )
        result = _apply_trial_params_to_node(node, {"n0-aggregation": "max"})
        self.assertEqual(result["layer"], "avgpool")
        self.assertEqual(
            result["_pushdown_aggregation"][
                "aggregation_function_aggregation_function"
            ],
            "max",
        )

    def test_apply_trial_params_to_node_mirrors_the_aggregation(self):
        node = RepresentationNode(
            node_id="n1",
            operation=AggregatedRepresentation,
            inputs=[],
            parameters={
                "aggregation_function_aggregation_function": "mean",
                "aggregation_function_pad_modality": True,
                "target_dimensions": 8,
            },
        )
        result = _apply_trial_params_to_node(node, {"n1-aggregation": "max"})
        self.assertEqual(result["aggregation_function_aggregation_function"], "max")
        self.assertNotIn("aggregation_function_pad_modality", result)
        self.assertEqual(result["target_dimensions"], 8)

    def test_is_window_operation_recognises_only_window_classes(self):
        for operation, expected in (
            (WindowAggregation, True),
            (AggregatedRepresentation, False),
            ("WindowAggregation", False),
        ):
            with self.subTest(operation=operation):
                self.assertEqual(_is_window_operation(operation), expected)

    def test_is_aggregated_representation_operation_recognises_only_its_classes(self):
        for operation, expected in (
            (AggregatedRepresentation, True),
            (WindowAggregation, False),
            (None, False),
        ):
            with self.subTest(operation=operation):
                self.assertEqual(
                    _is_aggregated_representation_operation(operation), expected
                )

    def test_operation_checks_fall_back_when_the_module_is_unavailable(self):
        # both checks import their base class inside the function
        checks = (
            (
                _is_window_operation,
                "systemds.scuro.representations.window_aggregation",
                WindowAggregation,
            ),
            (
                _is_aggregated_representation_operation,
                "systemds.scuro.representations.aggregated_representation",
                AggregatedRepresentation,
            ),
        )
        for check, module_name, operation in checks:
            with self.subTest(check=check.__name__):
                with patch.dict(sys.modules, {module_name: None}):
                    self.assertFalse(check(operation))

    def test_materialize_node_params_returns_the_input_outside_a_window(self):
        window_node = RepresentationNode(
            node_id="n0", operation=WindowAggregation, inputs=[]
        )
        leaf_node = RepresentationNode(node_id="n1", operation=None, inputs=[])
        for node, flat_params in ((leaf_node, {"window_size": 4}), (window_node, {})):
            with self.subTest(operation=node.operation):
                self.assertEqual(
                    _materialize_node_params(node, flat_params), flat_params
                )


class TestHyperparamResults(unittest.TestCase):
    """HyperparamResults stores what the tuner found. Reading it back rebuilds
    the DAG of a result with the parameters that scored best."""

    def setUp(self):
        self.task = SimpleNamespace(model=SimpleNamespace(name="TuningTask"))
        self.modality = SimpleNamespace(modality_id="audio_0")
        self.results = HyperparamResults([self.task], [self.modality])

    def _dag_with_one_operation(self):
        # the unimodal optimizer builds its DAGs through the same builder
        builder = RepresentationDAGBuilder()
        leaf_id = builder.create_leaf_node(self.modality.modality_id)
        operation_id = builder.create_operation_node(
            WindowAggregation, [leaf_id], {"window_size": 4}
        )
        return builder.build(operation_id), operation_id

    def _result(self, dag, best_params, mm_opt=False):
        return HyperparamResult(
            representation_name="WindowAggregation",
            best_params=best_params,
            best_score=0.8,
            all_results=[],
            tuning_time=0.1,
            modality_id=self.modality.modality_id,
            task_name=self.task.model.name,
            dag=dag,
            mm_opt=mm_opt,
        )

    def _store(self, results):
        self.results.results[self.task.model.name][self.modality.modality_id] = results

    def test_add_result_skips_a_missing_result(self):
        # a representation that could not be tuned arrives as None
        self.results.add_result([None])
        self.assertEqual(
            self.results.results[self.task.model.name][self.modality.modality_id], []
        )

    def test_add_result_stores_a_multimodal_result_under_its_own_key(self):
        dag, _ = self._dag_with_one_operation()
        self.results.setup_mm(optimize_unimodal=False)
        self.results.add_result([self._result(dag, {}, mm_opt=True)])
        self.assertEqual(
            len(self.results.results[self.task.model.name]["mm_results"]), 1
        )

    def test_setup_mm_replaces_the_results_with_a_multimodal_slot(self):
        self.results.setup_mm(optimize_unimodal=False)
        self.assertEqual(
            self.results.results, {self.task.model.name: {"mm_results": []}}
        )

    def test_setup_mm_keeps_the_results_when_the_unimodal_step_runs(self):
        self.results.setup_mm(optimize_unimodal=True)
        self.assertEqual(
            self.results.results,
            {self.task.model.name: {self.modality.modality_id: []}},
        )

    def test_get_k_best_dags_rebuilds_the_dag_with_the_best_parameters(self):
        dag, operation_id = self._dag_with_one_operation()
        self._store([self._result(dag, {f"{operation_id}-window_size": 8})])

        _, dags = self.results.get_k_best_dags(self.modality, self.task)

        operation_nodes = [node for node in dags[0].nodes if node.operation is not None]
        leaf_nodes = [node for node in dags[0].nodes if node.operation is None]
        self.assertEqual(operation_nodes[0].parameters["window_size"], 8)
        self.assertEqual(leaf_nodes[0].modality_id, self.modality.modality_id)
        self.assertEqual(dags[0].root_node_id, operation_nodes[0].node_id)
        # the stored result keeps the parameters it was evaluated with
        self.assertEqual(dag.get_node_by_id(operation_id).parameters["window_size"], 4)

    def test_get_k_best_dags_returns_one_dag_per_stored_result(self):
        first_dag, _ = self._dag_with_one_operation()
        second_dag, _ = self._dag_with_one_operation()
        stored = [self._result(first_dag, {}), self._result(second_dag, {})]
        self._store(stored)

        results, dags = self.results.get_k_best_dags(self.modality, self.task)

        self.assertIs(results, stored)
        self.assertEqual(len(dags), 2)

    def test_get_k_best_dags_returns_nothing_without_stored_results(self):
        self.assertEqual(
            self.results.get_k_best_dags(self.modality, self.task), ([], [])
        )

    def test_get_k_best_results_returns_the_last_output_of_every_dag(self):
        # executing a DAG returns one entry per node
        dag, _ = self._dag_with_one_operation()
        self._store([self._result(dag, {})])
        outputs = {"leaf": "raw modality", "operation": "representation"}

        with patch.object(RepresentationDag, "execute", return_value=outputs):
            _, representations = self.results.get_k_best_results(
                self.modality, self.task, "accuracy"
            )

        self.assertEqual(representations, ["representation"])


if __name__ == "__main__":
    unittest.main()
