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
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from systemds.scuro.drsearch.objectives import SEARCH_INFERENCE_TIME
from systemds.scuro.drsearch.representation_dag import RepresentationDAGBuilder
from systemds.scuro.drsearch.split_aware_evaluator import SplitAwareDAGEvaluator
from systemds.scuro.drsearch.task import INFERENCE_TIME
from systemds.scuro.drsearch.trial_result_cache import _clone
from systemds.scuro.drsearch.unimodal_optimizer import UnimodalResults
from systemds.scuro.modality.indexed import IndexedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.tfidf import TfIdf
from tests.scuro.data_generator import (
    ModalityRandomDataGenerator,
    TestDataLoader,
    TestTask,
)

NUM_INSTANCES = 20


def _text_modality():
    data, metadata = ModalityRandomDataGenerator().create_text_data(NUM_INSTANCES, 10)
    return UnimodalModality(
        TestDataLoader(
            np.arange(NUM_INSTANCES), None, ModalityType.TEXT, data, str, metadata
        )
    )


def _tfidf_dag(modality):
    builder = RepresentationDAGBuilder()
    return builder.build(
        builder.create_operation_node(
            TfIdf, [builder.create_leaf_node(modality.modality_id)], {}
        )
    )


def _slowed(method, seconds):
    def wrapper(*args, **kwargs):
        time.sleep(seconds)
        return method(*args, **kwargs)

    return wrapper


class TestInferenceCost(unittest.TestCase):
    def test_split_path_records_time_per_instance(self):
        text = _text_modality()
        task = TestTask("InferenceCostTask", "Test1", NUM_INSTANCES)

        evaluation = SplitAwareDAGEvaluator(_tfidf_dag(text), text, task).evaluate()

        val, test = evaluation.scores[1], evaluation.scores[2]
        self.assertEqual(len(val.time_per_instance), task.kfold)
        self.assertEqual(len(test.time_per_instance), task.kfold)
        self.assertAlmostEqual(
            val.average_scores[INFERENCE_TIME], np.mean(val.time_per_instance)
        )
        self.assertIn(INFERENCE_TIME, val.fold_scores())
        self.assertIn(INFERENCE_TIME, val.score_stds())

        timing = task.last_run_timing
        self.assertGreater(timing["representation_time_per_instance_s"], 0.0)
        self.assertGreater(timing["val_inference_time_per_instance_s"], 0.0)
        self.assertAlmostEqual(
            evaluation.inference_cost,
            (
                timing["representation_time_per_instance_s"]
                + timing["val_inference_time_per_instance_s"]
            )
            * len(task.test_indices),
        )

    def test_split_path_excludes_fit_time(self):
        text = _text_modality()
        task = TestTask("InferenceCostTask", "Test1", NUM_INSTANCES)

        with patch.object(TfIdf, "fit", _slowed(TfIdf.fit, 0.2)):
            evaluation = SplitAwareDAGEvaluator(_tfidf_dag(text), text, task).evaluate()

        n_transformed = task.kfold * (len(task.train_indices) + len(task.test_indices))
        transform_total = (
            task.last_run_timing["representation_time_per_instance_s"] * n_transformed
        )
        self.assertGreaterEqual(evaluation.representation_time, 0.2 * task.kfold)
        self.assertLess(transform_total, 0.2)

    def test_split_path_charges_leaf_loading(self):
        text = _text_modality()
        task = TestTask("InferenceCostTask", "Test1", NUM_INSTANCES)
        loader = text.data_loader

        with patch.object(loader, "load", _slowed(loader.load, 0.2)):
            SplitAwareDAGEvaluator(_tfidf_dag(text), text, task).evaluate()

        self.assertGreaterEqual(
            task.last_run_timing["representation_time_per_instance_s"],
            0.2 / NUM_INSTANCES,
        )

    def test_unimodal_front_is_retimed_serially(self):
        text = _text_modality()
        task = TestTask("InferenceCostTask", "Test1", NUM_INSTANCES)
        results = UnimodalResults(
            [text], [task], objectives=[("accuracy", "max"), (INFERENCE_TIME, "min")]
        )
        entry = SimpleNamespace(
            dag=_tfidf_dag(text), val_score={"accuracy": 0.5, INFERENCE_TIME: 100.0}
        )
        results.results[text.modality_id][task.model.name].append(entry)

        loader = text.data_loader
        with patch.object(loader, "load", _slowed(loader.load, 0.2)):
            front = results.get_pareto_front(text, task, retime_repeats=1)

        self.assertEqual(front, [entry])
        self.assertEqual(entry.val_score[SEARCH_INFERENCE_TIME], 100.0)
        self.assertGreaterEqual(entry.val_score[INFERENCE_TIME], 0.2 / NUM_INSTANCES)
        self.assertLess(entry.val_score[INFERENCE_TIME], 100.0)
        self.assertFalse(text.has_data())

    def test_task_run_records_time_per_instance(self):
        task = TestTask("InferenceCostTask", "Test1", NUM_INSTANCES)
        data = list(np.random.default_rng(0).random((NUM_INSTANCES, 4)))

        _, val, test = task.run(data, representation_time=2.0)

        representation_per_instance = 2.0 / NUM_INSTANCES
        self.assertEqual(len(val.time_per_instance), task.kfold)
        for fold, seconds in enumerate(val.time_per_instance):
            self.assertAlmostEqual(
                seconds,
                representation_per_instance
                + task.val_inference_time[fold] / len(task.cv_val_indices[fold]),
            )
        self.assertGreater(test.average_scores[INFERENCE_TIME], 0.1)

    def test_subset_prorates_and_cache_clone_keeps_transform_seconds(self):
        text = _text_modality()
        text.extract_raw_data()
        full = IndexedModality(text, range(NUM_INSTANCES), 2.0)

        part = full.subset(range(5))

        self.assertAlmostEqual(part.transform_seconds, 0.5)
        self.assertAlmostEqual(_clone(part).transform_seconds, 0.5)

    def test_leaf_batch_charges_each_representation_its_own_time(self):
        text = _text_modality()
        text.extract_raw_data()

        slow, fast = TfIdf(), BoW()
        with patch.object(TfIdf, "transform", _slowed(TfIdf.transform, 0.3)):
            results = text.apply_representations([slow, fast])

        self.assertGreaterEqual(results[slow.name].transform_time, 0.3)
        self.assertLess(results[fast.name].transform_time, 0.3)


if __name__ == "__main__":
    unittest.main()
