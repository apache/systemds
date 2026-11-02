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
from systemds.scuro.representations.color_histogram import ColorHistogram
from systemds.scuro.drsearch.operator_registry import Registry, load_builtin_operators
from systemds.scuro.drsearch.node_executor import ResultEntry
from systemds.scuro.drsearch.unimodal_optimizer import (
    UnimodalOptimizer,
    UnimodalResults,
    _lowest_fitted_key,
    evaluate_fitted_dags,
)
from systemds.scuro.representations.covarep_audio_features import ZeroCrossing

from systemds.scuro.representations.covarep_audio_features import (
    Spectral,
    RMSE,
    Pitch,
)
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.mfcc import MFCC
from systemds.scuro.representations.mlp_averaging import MLPAveraging
from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.tfidf import TfIdf
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.image_bind import ImageBind
from systemds.scuro.representations.text_context_with_indices import (
    OverlappingSplitIndices,
    SentenceBoundarySplitIndices,
)
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.window_aggregation import Window
from systemds.scuro.representations.concatenation import Concatenation
from tests.scuro.test_unimodal_representations import (
    PHYSIOLOGICAL_REPRESENTATIONS,
    TIMESERIES_REPRESENTATIONS,
)
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.dataloader.base_loader import LazyFileSequence
from tests.scuro.data_generator import (
    ModalityRandomDataGenerator,
    TestDataLoader,
    TestSVM,
    TestTask,
)
import copy

from systemds.scuro.drsearch.representation_dag import (
    CSEAwareDAGBuilder,
    RepresentationDag,
    RepresentationDAGBuilder,
    pushdown_aggregation,
)
from systemds.scuro.drsearch.split_aware_evaluator import (
    SplitAwareDAGEvaluator,
    dag_requires_fitting,
)
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.modality.type import ModalityType

from unittest.mock import patch

LIGHTWEIGHT_REGISTRY = {
    ModalityType.TEXT: [BoW, TfIdf],
    ModalityType.AUDIO: [MelSpectrogram, ZeroCrossing],
    ModalityType.VIDEO: [ResNet],
    ModalityType.IMAGE: [ColorHistogram],
    ModalityType.TIMESERIES: [],
    ModalityType.EMBEDDING: [],
}

#: Every registered representation that runs without downloading a pretrained
#: model. The transformer- and CNN-based ones (Bert, RoBERTa, CLIP, GloVe, X3D,
#: VGG19, Swin, Wav2Vec) are deliberately absent: they pull hundreds of MB over
#: the network, which is the same reason the video representation test in
#: test_unimodal_representations.py is commented out.
FULL_TEXT_REPRESENTATIONS = [BoW, TfIdf, W2V]
FULL_AUDIO_REPRESENTATIONS = [
    MFCC,
    MelSpectrogram,
    Spectrogram,
    Spectral,
    RMSE,
    Pitch,
    ZeroCrossing,
]
FULL_IMAGE_REPRESENTATIONS = [ColorHistogram]
FULL_TIMESERIES_REPRESENTATIONS = TIMESERIES_REPRESENTATIONS
FULL_PHYSIOLOGICAL_REPRESENTATIONS = PHYSIOLOGICAL_REPRESENTATIONS


def registry_for(modality_type, representations):
    """A registry holding `representations` for one modality and nothing else.

    Every modality type has to be present: the optimizer looks its modality up
    directly, and a partial dict would raise a KeyError rather than search an
    empty space.
    """
    registry = {m_type: [] for m_type in ModalityType}
    registry[modality_type] = representations
    return registry


class TestUnimodalRepresentationOptimizer(unittest.TestCase):
    data_generator = None
    num_instances = 0

    @classmethod
    def setUpClass(cls):
        cls.num_instances = 10
        cls.mods = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]

        cls.indices = np.array(range(cls.num_instances))

        cls.tasks = [
            TestTask("UnimodalRepresentationTask1", "Test1", cls.num_instances),
        ]

    def test_unimodal_optimizer_for_text_modality(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances, 10
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        self.optimize_unimodal_representation_for_modality([text])

    def test_fitted_text_leaves_are_refit_per_fold_in_fusion(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances, 10
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        task = self.tasks[0]
        results = UnimodalResults([text], [task], k=2)
        for accuracy, representation in ((0.9, BoW), (0.8, TfIdf)):
            builder = RepresentationDAGBuilder()
            leaf = builder.create_leaf_node(text.modality_id)
            dag = builder.build(
                builder.create_operation_node(representation, [leaf], {})
            )
            results.results[text.modality_id][task.model.name].append(
                SimpleNamespace(
                    dag=dag,
                    val_score={"accuracy": accuracy},
                    representation_time=0.0,
                    task_time=0.0,
                )
            )
        _, leaves = results.get_k_best_results(text, task, "accuracy")
        self.assertEqual(len(leaves), 2)
        self.assertTrue(all(leaf.fold_source for leaf in leaves))

        builder = RepresentationDAGBuilder()
        fused = builder.build(
            builder.create_operation_node(
                Concatenation,
                [
                    builder.create_leaf_node(text.modality_id, 0),
                    builder.create_leaf_node(text.modality_id, 1),
                ],
                {},
            )
        )
        self.assertTrue(dag_requires_fitting(fused, leaves))

        fit_sets = []
        fits = {cls: cls.fit for cls in (BoW, TfIdf)}

        def recording_fit(cls):
            def fit(op, inputs, labels=None):
                fit_sets.append(set(inputs.instance_ids))
                return fits[cls](op, inputs, labels=labels)

            return fit

        with patch.object(BoW, "fit", recording_fit(BoW)), patch.object(
            TfIdf, "fit", recording_fit(TfIdf)
        ):
            SplitAwareDAGEvaluator(fused, leaves, task).evaluate()
            SplitAwareDAGEvaluator(fused, leaves, task).evaluate()

        allowed = [set(split.fit_indices) for split in task.evaluation_splits]
        self.assertEqual(len(fit_sets), 2 * len(allowed))
        self.assertTrue(all(fit_set in allowed for fit_set in fit_sets))

    def test_k_best_dags_decode_a_lazy_leaf_once(self):
        images, image_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances
        )
        task = self.tasks[0]

        def k_best(decoded_bytes):
            loader = TestDataLoader(
                self.indices, None, ModalityType.IMAGE, images, np.uint8, image_md
            )
            loader.estimate_decoded_bytes = lambda metadata=None: decoded_bytes
            image = UnimodalModality(loader)
            decoded = []

            def decode(name):
                decoded.append(name)
                return images[int(name)].copy()

            lazy = LazyFileSequence([str(i) for i in self.indices], decode)
            image._data, image.metadata = lazy, image_md
            results = UnimodalResults([image], [task], k=2)
            for accuracy, bins in ((0.9, 16), (0.8, 32)):
                builder = RepresentationDAGBuilder()
                leaf = builder.create_leaf_node(image.modality_id)
                dag = builder.build(
                    builder.create_operation_node(
                        ColorHistogram, [leaf], {"bins": bins}
                    )
                )
                results.results[image.modality_id][task.model.name].append(
                    SimpleNamespace(dag=dag, val_score={"accuracy": accuracy})
                )
            _, cache = results.get_k_best_results(image, task, "accuracy")
            self.assertIs(image.data, lazy)
            return len(decoded), [np.asarray(c.data) for c in cache]

        lazy_decodes, lazy_outputs = k_best(None)
        once_decodes, once_outputs = k_best(1)
        self.assertEqual(lazy_decodes, 2 * self.num_instances)
        self.assertEqual(once_decodes, self.num_instances)
        for lazy_output, once_output in zip(lazy_outputs, once_outputs):
            np.testing.assert_array_equal(lazy_output, once_output)

    def _text_and_two_tasks(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances, 10
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        second = copy.copy(self.tasks[0])
        second.model = TestSVM("Test2")
        return text, [self.tasks[0], second]

    def test_fitted_dags_fit_once_per_fold_for_all_dags_and_tasks(self):
        text, tasks = self._text_and_two_tasks()
        builder = RepresentationDAGBuilder()
        tfidf = builder.create_operation_node(
            TfIdf, [builder.create_leaf_node(text.modality_id)], {}
        )
        plain = builder.build(tfidf, dag_id=0)
        reduced = builder.build(
            builder.create_operation_node(
                MLPAveraging, [tfidf], MLPAveraging().get_current_parameters()
            ),
            dag_id=1,
        )
        self.assertEqual(_lowest_fitted_key(plain), _lowest_fitted_key(reduced))

        fit_sets, inputs = [], {}
        tfidf_fit, svm_fit = TfIdf.fit, TestSVM.fit

        def recording_fit(op, data, labels=None):
            fit_sets.append(set(data.instance_ids))
            return tfidf_fit(op, data, labels=labels)

        def recording_svm(model, X, y, X_test, y_test):
            inputs.setdefault(model.name, []).append(np.asarray(X))
            return svm_fit(model, X, y, X_test, y_test)

        with patch.object(TfIdf, "fit", recording_fit), patch.object(
            TestSVM, "fit", recording_svm
        ):
            results = evaluate_fitted_dags([plain, reduced], text, tasks)
            shared_inputs, inputs = inputs, {}
            for dag in (plain, reduced):
                SplitAwareDAGEvaluator(dag, text, tasks[0]).evaluate()

        folds = [set(split.fit_indices) for split in tasks[0].evaluation_splits]
        self.assertEqual(fit_sets[: len(folds)], folds)
        self.assertEqual(len(fit_sets), 3 * len(folds))
        self.assertEqual(
            [(dag_id, task_idx) for dag_id, task_idx, _, _ in results],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        self.assertTrue(all(r[2].representation_time > 0 for r in results))
        reference = inputs["Test1"]
        for name in ("Test1", "Test2"):
            self.assertEqual(len(shared_inputs[name]), len(reference))
            for shared, fresh in zip(shared_inputs[name], reference):
                np.testing.assert_array_equal(shared, fresh)

    def test_fitted_dags_are_evaluated_on_executor_workers(self):
        text, tasks = self._text_and_two_tasks()
        load_builtin_operators((ModalityType.TEXT, ModalityType.EMBEDDING))
        with patch.object(
            Registry, "_representations", registry_for(ModalityType.TEXT, [TfIdf])
        ), patch(
            "systemds.scuro.drsearch.unimodal_optimizer.evaluate_fitted_dags",
            side_effect=AssertionError("fitted DAGs evaluated in the driver"),
        ):
            optimizer = UnimodalOptimizer(
                [text], tasks, False, max_num_workers=2, enable_checkpointing=False
            )
            optimizer.optimize()

        fitted_ids = [
            {
                entry.dag.dag_id
                for entry in optimizer.operator_performance.results[text.modality_id][
                    task.model.name
                ]
                if dag_requires_fitting(entry.dag)
            }
            for task in tasks
        ]
        self.assertTrue(fitted_ids[0])
        self.assertEqual(fitted_ids[0], fitted_ids[1])

    def test_k_best_tasks_share_an_unsupervised_leaf_fit(self):
        text, tasks = self._text_and_two_tasks()
        results = UnimodalResults([text], tasks, k=1)
        builder = RepresentationDAGBuilder()
        leaf = builder.create_leaf_node(text.modality_id)
        dag = builder.build(builder.create_operation_node(TfIdf, [leaf], {}))
        for task in tasks:
            results.results[text.modality_id][task.model.name].append(
                SimpleNamespace(dag=dag, val_score={"accuracy": 0.9})
            )
        fits = []
        fit_transform = SplitAwareDAGEvaluator.fit_transform

        def recording(evaluator, *args):
            fits.append(evaluator.task.model.name)
            return fit_transform(evaluator, *args)

        with patch.object(SplitAwareDAGEvaluator, "fit_transform", recording):
            leaves = [
                results.get_k_best_results(text, task, "accuracy")[1][0]
                for task in tasks
            ]

        self.assertEqual(fits, ["Test1"])
        self.assertIsNot(leaves[0], leaves[1])
        self.assertEqual(leaves[1].fold_source, (dag, text))
        np.testing.assert_array_equal(
            np.asarray(leaves[0].data), np.asarray(leaves[1].data)
        )

    def test_robust_results_ignore_non_finite_scores(self):
        modality = SimpleNamespace(modality_id="modality")
        task = SimpleNamespace(model=SimpleNamespace(name="task"))
        results = UnimodalResults([modality], [task], k=2)
        scores = [0.8, np.nan, np.inf, -np.inf, None, 0.6]
        entries = [
            ResultEntry(
                val_score=None if score is None else {"accuracy": score},
                dag=SimpleNamespace(
                    nodes=[SimpleNamespace(operation=UnimodalOptimizer)]
                ),
            )
            for score in scores
        ]
        results.results[modality.modality_id][task.model.name] = entries

        robust, indices = results.get_k_most_robust_results(
            modality, task, "accuracy", one_se_parsimony=False
        )

        self.assertEqual(robust, [entries[0], entries[5]])
        self.assertEqual(indices, [0, 5])
        self.assertTrue(all(np.isfinite(entry.robustness_score) for entry in robust))

    def test_robust_results_penalise_fold_variation(self):
        modality = SimpleNamespace(modality_id="modality")
        task = SimpleNamespace(model=SimpleNamespace(name="task"))
        results = UnimodalResults([modality], [task], k=2)
        entries = [
            ResultEntry(
                val_score={"accuracy": 0.90},
                val_fold_scores={"accuracy": [0.70, 0.80, 0.90, 1.00, 1.10]},
                dag=SimpleNamespace(
                    nodes=[SimpleNamespace(operation=UnimodalOptimizer)]
                ),
            ),
            ResultEntry(
                val_score={"accuracy": 0.86},
                val_fold_scores={"accuracy": [0.85, 0.86, 0.87, 0.85, 0.87]},
                dag=SimpleNamespace(
                    nodes=[SimpleNamespace(operation=UnimodalOptimizer)]
                ),
            ),
        ]
        results.results[modality.modality_id][task.model.name] = entries

        robust, indices = results.get_k_most_robust_results(
            modality, task, "accuracy", one_se_parsimony=False
        )

        self.assertEqual(robust, [entries[1], entries[0]])
        self.assertEqual(indices, [1, 0])
        self.assertAlmostEqual(
            entries[0].robustness_score,
            0.90 - np.std(entries[0].val_fold_scores["accuracy"], ddof=1),
        )
        self.assertAlmostEqual(
            entries[1].robustness_score,
            0.86 - np.std(entries[1].val_fold_scores["accuracy"], ddof=1),
        )

    def test_bow_and_tfidf_require_dimensionality_reduction_before_task(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances, 10
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )

        dimensionality_reduction_operators = {ModalityType.EMBEDDING: [MLPAveraging]}
        for representation in (BoW, TfIdf):
            with self.subTest(representation=representation.__name__), patch.object(
                Registry,
                "_representations",
                registry_for(ModalityType.TEXT, [representation]),
            ), patch.object(
                Registry,
                "_dimensionality_reduction_operators",
                dimensionality_reduction_operators,
            ):
                optimizer = UnimodalOptimizer(
                    [text], self.tasks, False, enable_checkpointing=False
                )
                _, _, task_dags = optimizer._build_execution_dags_for_modality(text)

                self.assertGreater(len(task_dags), 0)
                for dag in task_dags:
                    task_node = dag.get_node_by_id(dag.root_node_id)
                    task_input = dag.get_node_by_id(task_node.inputs[0])
                    self.assertIs(task_input.operation, MLPAveraging)

    def test_dimensionality_reduction_is_not_applied_to_sequence_outputs(self):
        video_data, video_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 10, 10, 10
        )
        video = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.VIDEO, video_data, np.float32, video_md
            )
        )
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, 4000
        )
        audio = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )
        dimensionality_reduction_operators = {ModalityType.EMBEDDING: [MLPAveraging]}
        for modality, representations in ((video, [ResNet]), (audio, [Spectral])):
            with self.subTest(modality=modality.modality_type.name), patch.object(
                Registry,
                "_representations",
                registry_for(modality.modality_type, representations),
            ), patch.object(
                Registry,
                "_dimensionality_reduction_operators",
                dimensionality_reduction_operators,
            ):
                optimizer = UnimodalOptimizer(
                    [modality], self.tasks, False, enable_checkpointing=False
                )
                dags, _, _ = optimizer._build_execution_dags_for_modality(modality)
                stats = {}
                for dag in dags:
                    stats.update(optimizer._node_output_stats(dag))
                reduce_input_ranks = [
                    len(stats[node.inputs[0]].output_shape)
                    for dag in dags
                    for node in dag.nodes
                    if node.operation is MLPAveraging
                ]
                self.assertEqual([rank for rank in reduce_input_ranks if rank > 1], [])
                self.assertTrue(
                    any(
                        p["operation"] == "MLPAveraging" and "sequence" in p["reason"]
                        for p in optimizer.pruned
                    )
                )

    def _window_inputs(self, modality, representations):
        """(input operation, input rank) of every window node in the search."""
        with patch.object(
            Registry,
            "_representations",
            registry_for(modality.modality_type, representations),
        ):
            optimizer = UnimodalOptimizer(
                [modality], self.tasks, False, enable_checkpointing=False
            )
            dags, _, _ = optimizer._build_execution_dags_for_modality(modality)
            stats = {}
            for dag in dags:
                stats.update(optimizer._node_output_stats(dag))
            inputs = []
            for dag in dags:
                for node in dag.nodes:
                    if node.operation is None or not issubclass(node.operation, Window):
                        continue
                    source = dag.get_node_by_id(node.inputs[0])
                    inputs.append(
                        (
                            source.operation,
                            len(stats[source.node_id].output_shape),
                        )
                    )
            return optimizer, inputs

    def test_windows_are_not_applied_to_rank_one_representation_outputs(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances, 10
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, 4000
        )
        audio = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )
        # W2V pools to one vector per instance; Concatenation flattens the
        # (frames, features) outputs of Spectral and ZeroCrossing to rank 1.
        for modality, representations in (
            (text, [W2V]),
            (audio, [Spectral, ZeroCrossing]),
        ):
            with self.subTest(modality=modality.modality_type.name):
                optimizer, inputs = self._window_inputs(modality, representations)
                representation_inputs = [
                    (operation, rank)
                    for operation, rank in inputs
                    if operation is not None
                ]
                self.assertEqual(representation_inputs, [])
                self.assertTrue(
                    any("no time axis" in p["reason"] for p in optimizer.pruned)
                )

    def test_windows_are_kept_for_representations_with_a_time_axis(self):
        video_data, video_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 10, 10, 10
        )
        video = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.VIDEO, video_data, np.float32, video_md
            )
        )
        _, inputs = self._window_inputs(video, [ResNet])
        self.assertGreater(len(inputs), 0)
        for operation, rank in inputs:
            self.assertIs(operation, ResNet)
            self.assertGreaterEqual(rank, 2)

    def test_unimodal_optimizer_for_image_modality(self):
        image_data, image_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 1, 10, 10
        )
        image = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.IMAGE, image_data, np.float32, image_md
            )
        )
        self.optimize_unimodal_representation_for_modality([image])

    def test_unimodal_optimizer_for_multiple_modalities(self):
        image_data, image_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 1, 10, 10
        )
        image = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.IMAGE, image_data, np.float32, image_md
            )
        )
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        self.optimize_unimodal_representation_for_modality([text, image])

    def test_unimodal_optimizer_for_audio_modality(self):
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, 3000
        )
        audio = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )

        self.optimize_unimodal_representation_for_modality([audio])

    def test_unimodal_optimizer_for_video_modality(self):
        video_data, video_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 10, 10, 10
        )
        video = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.VIDEO, video_data, np.float32, video_md
            )
        )
        self.optimize_unimodal_representation_for_modality([video])

    # ------------------------------------------------------------------
    # Every registered representation, run through the optimizer
    # ------------------------------------------------------------------
    #
    # The tests above check that the optimizer runs at all. These check that no
    # individual representation breaks it: a rep whose get_output_stats,
    # preconditions or transform disagree with what the executor expects takes
    # the whole search down, and with a two-representation registry that would
    # never surface.

    def _optimize_with_registry(self, modality, registry):
        with patch.object(Registry, "_representations", registry):
            Registry()
            unimodal_optimizer = UnimodalOptimizer(
                [modality],
                self.tasks,
                False,
                k=1,
                max_num_workers=1,
                enable_checkpointing=False,
            )
            unimodal_optimizer.optimize()

            self.assertIn(
                modality.modality_id,
                unimodal_optimizer.operator_performance.modality_ids,
            )
            result, _ = unimodal_optimizer.operator_performance.get_k_best_results(
                modality, self.tasks[0], "accuracy"
            )
            self.assertEqual(len(result), 1)
            return unimodal_optimizer

    def test_unimodal_optimizer_with_all_text_representations(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances, 10
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        self._optimize_with_registry(
            text, registry_for(ModalityType.TEXT, FULL_TEXT_REPRESENTATIONS)
        )

    def test_unimodal_optimizer_with_all_audio_representations(self):
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, 4000
        )
        audio = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )
        self._optimize_with_registry(
            audio, registry_for(ModalityType.AUDIO, FULL_AUDIO_REPRESENTATIONS)
        )

    def test_unimodal_optimizer_with_all_image_representations(self):
        image_data, image_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 1, 10, 10
        )
        image = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.IMAGE, image_data, np.float32, image_md
            )
        )
        self._optimize_with_registry(
            image, registry_for(ModalityType.IMAGE, FULL_IMAGE_REPRESENTATIONS)
        )

    def test_unimodal_optimizer_with_all_timeseries_representations(self):
        ts_data, ts_md = ModalityRandomDataGenerator().create_timeseries_data(
            self.num_instances, 256
        )
        timeseries = UnimodalModality(
            TestDataLoader(
                self.indices,
                None,
                ModalityType.TIMESERIES,
                ts_data,
                np.float32,
                ts_md,
            )
        )
        optimizer = self._optimize_with_registry(
            timeseries,
            registry_for(ModalityType.TIMESERIES, FULL_TIMESERIES_REPRESENTATIONS),
        )
        # A search over windowed timeseries always proposes some configurations
        # the input cannot express (a lag longer than the window, a moment on a
        # two-sample window). Those must be pruned up front, not executed.
        self.assertGreater(len(optimizer.pruned), 0)

    def test_unimodal_optimizer_with_all_physiological_representations(self):
        data, md = ModalityRandomDataGenerator().create_physiological_data(
            self.num_instances, 2000, kind="ecg", fs=500.0
        )
        physiological = UnimodalModality(
            TestDataLoader(
                self.indices,
                None,
                ModalityType.PHYSIOLOGICAL,
                data,
                np.float32,
                md,
            )
        )
        self._optimize_with_registry(
            physiological,
            registry_for(
                ModalityType.PHYSIOLOGICAL, FULL_PHYSIOLOGICAL_REPRESENTATIONS
            ),
        )

    def test_aggregation_pushdown_preserves_dag_id_and_bert_node_parameters(self):
        builder = CSEAwareDAGBuilder()
        modality_id = "test_modality_agg_pushdown"
        leaf_id = builder.create_leaf_node(modality_id)

        bert = Bert()
        bert_id = builder.create_operation_node(
            Bert, [leaf_id], bert.get_current_parameters()
        )

        agg = AggregatedRepresentation(target_dimensions=1)
        agg_id = builder.create_operation_node(
            AggregatedRepresentation,
            [bert_id],
            agg.get_current_parameters(),
        )

        expected_dag_id = 1001
        dag = RepresentationDag(
            nodes=copy.deepcopy(builder.global_nodes),
            root_node_id=agg_id,
            dag_id=expected_dag_id,
        )

        by_id = {n.node_id: n for n in dag.nodes}
        self.assertEqual(len(dag.nodes), 3)
        self.assertEqual(dag.dag_id, expected_dag_id)
        self.assertEqual(dag.root_node_id, agg_id)

        self.assertEqual(by_id[leaf_id].inputs, [])
        self.assertEqual(by_id[bert_id].inputs, [leaf_id])
        self.assertEqual(by_id[agg_id].inputs, [bert_id])
        self.assertIs(by_id[bert_id].operation, Bert)
        self.assertIs(by_id[agg_id].operation, AggregatedRepresentation)

        bert_params_before = copy.deepcopy(by_id[bert_id].parameters)
        agg_params_snapshot = copy.deepcopy(by_id[agg_id].parameters)
        self.assertNotIn("_pushdown_aggregation", bert_params_before)

        pushdown_aggregation([dag])

        self.assertEqual(dag.dag_id, expected_dag_id)
        self.assertEqual(dag.root_node_id, agg_id)
        self.assertEqual(len(dag.nodes), 2)
        self.assertIsNone(dag.get_node_by_id(bert_id))

        bert_after = dag.get_node_by_id(agg_id)
        self.assertIsNotNone(bert_after)
        self.assertIs(bert_after.operation, Bert)
        self.assertEqual(bert_after.inputs, [leaf_id])
        self.assertIn("_pushdown_aggregation", bert_after.parameters)
        self.assertEqual(
            bert_after.parameters["_pushdown_aggregation"],
            agg_params_snapshot,
        )
        remaining = {
            k: v
            for k, v in bert_after.parameters.items()
            if k != "_pushdown_aggregation"
        }
        self.assertEqual(remaining, bert_params_before)

    def optimize_unimodal_representation_for_modality(self, modalities):
        with patch.object(
            Registry,
            "_representations",
            LIGHTWEIGHT_REGISTRY,
        ):
            registry = Registry()

            unimodal_optimizer = UnimodalOptimizer(
                modalities,
                self.tasks,
                False,
                k=1,
                max_num_workers=1,
                enable_checkpointing=False,
            )
            unimodal_optimizer.optimize()
            for modality in modalities:
                assert (
                    modality.modality_id
                    in unimodal_optimizer.operator_performance.modality_ids
                )

            assert len(unimodal_optimizer.operator_performance.task_names) == 1
            result, cached = unimodal_optimizer.operator_performance.get_k_best_results(
                modalities[0], self.tasks[0], "accuracy"
            )
            assert len(result) == 1


class TestSelectionStrategies(unittest.TestCase):
    """The three selection strategies must genuinely disagree.

    They are only worth carrying separately through HPO and fusion if they can
    pick different winners out of the same pool, so the fixture is built so
    each strategy has a different answer: A wins on raw score, B on robustness
    (low fold variation, small DAG), and B again once runtime counts, with A
    demoted for being slow.
    """

    @staticmethod
    def _entry(accuracy, std, runtime, n_nodes):
        dag = SimpleNamespace(
            nodes=[SimpleNamespace(operation=object()) for _ in range(n_nodes)]
        )
        return SimpleNamespace(
            dag=dag,
            val_score={"accuracy": accuracy},
            val_fold_scores={"accuracy": [accuracy - std, accuracy + std]},
            val_score_stds={"accuracy": std},
            representation_time=runtime,
            task_time=0.0,
        )

    def setUp(self):
        self.modality = SimpleNamespace(modality_id="m1")
        self.task = SimpleNamespace(model=SimpleNamespace(name="t1"))
        self.entries = {
            "noisy_slow": self._entry(0.90, 0.10, 100.0, 5),
            "stable_fast": self._entry(0.88, 0.01, 10.0, 2),
            "weak_fastest": self._entry(0.85, 0.05, 1.0, 1),
        }
        self.results = UnimodalResults([self.modality], [self.task], k=3)
        self.results.results["m1"]["t1"] = list(self.entries.values())

    def _winner(self, strategy, **kwargs):
        view = self.results.with_selection_strategy(strategy, **kwargs)
        ranked, _ = view.get_k_best_results(
            self.modality, self.task, "accuracy", cache_needed=False
        )
        return ranked[0]

    def test_performance_takes_the_validation_argmax(self):
        self.assertIs(self._winner("performance"), self.entries["noisy_slow"])

    def test_robustness_penalises_fold_variation(self):
        self.assertIs(self._winner("robustness"), self.entries["stable_fast"])

    def test_tradeoff_weighs_runtime_against_accuracy(self):
        self.assertIs(self._winner("tradeoff"), self.entries["stable_fast"])

    def test_tradeoff_with_all_weight_on_accuracy_matches_performance(self):
        self.assertIs(
            self._winner("tradeoff", tradeoff_weights=(1.0, 0.0)),
            self.entries["noisy_slow"],
        )

    def test_views_share_the_candidate_pool_and_leave_the_source_alone(self):
        view = self.results.with_selection_strategy("robustness")
        self.assertIs(view.results, self.results.results)
        self.assertIs(view.cache, self.results.cache)
        self.assertEqual(self.results.selection_strategy, "performance")

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            self.results.with_selection_strategy("cheapest")
