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


import os
import pickle
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from systemds.scuro.representations.color_histogram import ColorHistogram
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.node_executor import ResultEntry
from systemds.scuro.drsearch.unimodal_optimizer import (
    UnimodalOptimizer,
    UnimodalResults,
    get_dag_by_id,
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
from systemds.scuro.representations.word2vec import W2V
from tests.scuro.test_unimodal_representations import (
    PHYSIOLOGICAL_REPRESENTATIONS,
    TIMESERIES_REPRESENTATIONS,
)
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from tests.scuro.data_generator import (
    ModalityRandomDataGenerator,
    TestDataLoader,
    TestTask,
)
import copy

from systemds.scuro.drsearch.representation_dag import (
    CSEAwareDAGBuilder,
    RepresentationDag,
    pushdown_aggregation,
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

    # (label, [(modality type, generator keyword arguments)]). A set with two
    # entries is passed to the optimizer as one multi-modality search, because
    # optimize_unimodal_representation_for_modality loops over the list.
    MODALITY_SETS = [
        ("text", [(ModalityType.TEXT, {})]),
        ("image", [(ModalityType.IMAGE, {})]),
        ("audio", [(ModalityType.AUDIO, {})]),
        ("video", [(ModalityType.VIDEO, {"num_frames": 10})]),
        (
            "text+image",
            [(ModalityType.TEXT, {"num_sentences": 1}), (ModalityType.IMAGE, {})],
        ),
    ]

    def _create_modality(self, modality_type, num_sentences=10, num_frames=1):
        generator = ModalityRandomDataGenerator()
        if modality_type is ModalityType.TEXT:
            data, metadata = generator.create_text_data(
                self.num_instances, num_sentences
            )
            data_type = str
        elif modality_type is ModalityType.AUDIO:
            data, metadata = generator.create_audio_data(self.num_instances, 3000)
            data_type = np.float32
        else:
            # IMAGE and VIDEO use the same generator. The number of frames is
            # the difference between them.
            data, metadata = generator.create_visual_modality(
                self.num_instances, num_frames, 10, 10
            )
            data_type = np.float32

        return UnimodalModality(
            TestDataLoader(self.indices, None, modality_type, data, data_type, metadata)
        )

    def test_unimodal_optimizer_per_modality_set(self):
        for label, modality_specs in self.MODALITY_SETS:
            with self.subTest(modalities=label):
                self.optimize_unimodal_representation_for_modality(
                    [
                        self._create_modality(modality_type, **kwargs)
                        for modality_type, kwargs in modality_specs
                    ]
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


class TestUnimodalOptimizerPersistence(unittest.TestCase):
    """store_results writes the search results to disk so that a long search
    can be resumed or inspected afterwards. The files are written from the
    result container, so no search has to run."""

    # TestTask stratifies its train and validation split, so it needs at least
    # one instance per class in each part.
    num_instances = 10

    def setUp(self):
        self.result_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.result_path)
        indices = np.array(range(self.num_instances))
        data, metadata = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, 200
        )
        self.modality = UnimodalModality(
            TestDataLoader(
                indices, None, ModalityType.AUDIO, data, np.float32, metadata
            )
        )
        self.task = TestTask("PersistenceTask", "Test1", self.num_instances)
        self.optimizer = UnimodalOptimizer(
            [self.modality],
            [self.task],
            result_path=self.result_path,
            enable_checkpointing=False,
        )

    def _read(self, file_name):
        with open(os.path.join(self.result_path, file_name), "rb") as f:
            return pickle.load(f)

    def test_store_results_writes_the_results_and_the_execution_statistics(self):
        self.optimizer.store_results("results.pkl")

        self.assertEqual(
            sorted(os.listdir(self.result_path)),
            ["results.pkl", "results_exec_stats.pkl"],
        )
        self.assertEqual(
            self._read("results.pkl"), self.optimizer.operator_performance.results
        )
        self.assertEqual(
            set(self._read("results_exec_stats.pkl")),
            {
                "worker_stats",
                "node_stats",
                "reuse_stats",
                "wall_clock_s",
                "search_start_unix",
                "max_num_workers",
            },
        )

    def test_store_results_names_the_file_after_the_optimizer_and_the_time(self):
        self.optimizer.store_results()

        written = sorted(os.listdir(self.result_path))
        self.assertEqual(len(written), 2)
        for name in written:
            self.assertTrue(name.startswith("unimodal_optimizer"))
            self.assertTrue(name.endswith(".pkl"))

    def test_store_results_appends_the_statistics_suffix_without_a_pkl_ending(self):
        # The statistics file name is derived by replacing ".pkl". Without that
        # ending the name stays unchanged, so the suffix is appended.
        self.optimizer.store_results("results")

        self.assertEqual(
            sorted(os.listdir(self.result_path)),
            ["results", "results_exec_stats.pkl"],
        )

    def test_results_survive_a_store_and_load_round_trip(self):
        modality_id = self.modality.modality_id
        task_name = self.task.model.name
        entry = ResultEntry(
            val_score={"accuracy": 0.75}, representation_time=1.0, task_time=2.0
        )
        self.optimizer.operator_performance.results[modality_id][task_name] = [entry]

        self.optimizer.store_results("results.pkl")
        self.optimizer.operator_performance.results[modality_id][task_name] = []
        self.optimizer.load_results(os.path.join(self.result_path, "results.pkl"))

        restored = self.optimizer.operator_performance.results[modality_id][task_name]
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].val_score, {"accuracy": 0.75})

    def test_count_results_by_modality_counts_the_first_task_of_every_modality(self):
        # The count drives the checkpoint progress, where every task evaluates
        # the same representations, so one task stands for all of them.
        counts = self.optimizer._count_results_by_modality(
            {"audio": {"t1": [1, 2], "t2": [3]}, "text": {"t1": []}}
        )

        self.assertEqual(counts, {"audio": 2, "text": 0})

    def test_resume_from_checkpoint_restores_a_stored_result_set(self):
        restored = {"audio": {"t1": ["entry"]}}
        with patch.object(
            self.optimizer._checkpoint_manager,
            "resume_from_checkpoint",
            return_value=(restored, None, None),
        ):
            self.optimizer.resume_from_checkpoint()

        self.assertEqual(self.optimizer.operator_performance.results, restored)

    def test_resume_from_checkpoint_keeps_the_results_without_a_checkpoint(self):
        current = self.optimizer.operator_performance.results
        with patch.object(
            self.optimizer._checkpoint_manager,
            "resume_from_checkpoint",
            return_value=None,
        ):
            self.optimizer.resume_from_checkpoint()

        self.assertIs(self.optimizer.operator_performance.results, current)


class TestUnimodalResultsReadout(unittest.TestCase):
    """UnimodalResults holds the scores of every evaluated representation and a
    cache of the data the best ones produced. get_k_best_results returns both.
    The entries are written straight into the container, so no search has to
    run."""

    num_instances = 10

    def setUp(self):
        indices = np.array(range(self.num_instances))
        data, metadata = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, 200
        )
        self.modality = UnimodalModality(
            TestDataLoader(
                indices, None, ModalityType.AUDIO, data, np.float32, metadata
            )
        )
        self.task = TestTask("ReadoutTask", "Test1", self.num_instances)
        self.results = UnimodalResults(
            [self.modality], [self.task], k=2, metric_name="accuracy"
        )
        self.modality_id = self.modality.modality_id
        self.task_name = self.task.model.name

    def _entry(self, accuracy, dag=None):
        return ResultEntry(
            val_score={"accuracy": accuracy},
            representation_time=1.0,
            task_time=1.0,
            dag=dag,
        )

    def _fill(self, accuracies):
        entries = [self._entry(a) for a in accuracies]
        self.results.results[self.modality_id][self.task_name] = entries
        return entries

    def test_get_k_best_results_returns_the_k_highest_scores(self):
        self._fill([0.5, 0.9, 0.7, 0.3])

        best, _ = self.results.get_k_best_results(
            self.modality, self.task, "accuracy", cache_needed=False
        )

        self.assertEqual([entry.val_score["accuracy"] for entry in best], [0.9, 0.7])

    def test_get_k_best_results_uses_the_cache_when_it_holds_entries(self):
        self._fill([0.9, 0.7])
        self.results.cache[self.modality_id][self.task_name] = ["first", "second"]

        _, cache = self.results.get_k_best_results(self.modality, self.task, "accuracy")

        self.assertEqual(cache, ["first", "second"])

    def test_get_k_best_results_executes_the_dag_when_the_cache_is_empty(self):
        # load_results restores the scores but not the cached data, so the
        # cache is empty after reading a file. The dag of every returned entry
        # is executed to rebuild it.
        executed = []

        class RecordingDag:
            def __init__(self, label):
                self.label = label

            def execute(self, modalities):
                executed.append(self.label)
                return self.label

        entries = [
            self._entry(0.9, dag=RecordingDag("best")),
            self._entry(0.7, dag=RecordingDag("second")),
        ]
        self.results.results[self.modality_id][self.task_name] = entries
        self.results.cache[self.modality_id][self.task_name] = []

        _, cache = self.results.get_k_best_results(self.modality, self.task, "accuracy")

        self.assertEqual(executed, ["best", "second"])
        self.assertEqual(cache, ["best", "second"])

    def test_get_k_best_results_returns_nothing_without_entries(self):
        best, cache = self.results.get_k_best_results(
            self.modality, self.task, "accuracy", cache_needed=False
        )

        self.assertEqual(best, [])
        self.assertEqual(cache, [])

    def test_get_dag_by_id_finds_the_matching_dag(self):
        dags = [SimpleNamespace(dag_id=1), SimpleNamespace(dag_id=7)]

        self.assertIs(get_dag_by_id(dags, 7), dags[1])

    def test_get_dag_by_id_returns_none_for_an_unknown_id(self):
        dags = [SimpleNamespace(dag_id=1)]

        self.assertIsNone(get_dag_by_id(dags, 99))
