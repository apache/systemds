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

import numpy as np

from systemds.scuro.drsearch.evaluation_split import EvaluationSplit
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDAGBuilder,
    RepresentationNode,
)
from systemds.scuro.drsearch.split_aware_evaluator import SplitAwareDAGEvaluator
from systemds.scuro.modality.indexed import IndexedModality
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.mlp_averaging import MLPAveraging
from systemds.scuro.representations.representation import FitRequirement
from systemds.scuro.representations.tfidf import TfIdf
from tests.scuro.data_generator import (
    ModalityRandomDataGenerator,
    TestDataLoader,
    TestTask,
)


# fits like TfIdf, but declares that it needs labels
class _SupervisedTfIdf(TfIdf):
    fit_requirement = FitRequirement.SUPERVISED


def _text_modality(num_instances):
    data, metadata = ModalityRandomDataGenerator().create_text_data(num_instances, 10)
    return UnimodalModality(
        TestDataLoader(
            np.array(range(num_instances)),
            None,
            ModalityType.TEXT,
            data,
            str,
            metadata,
        )
    )


def _single_operation_dag(modality, operation):
    builder = RepresentationDAGBuilder()
    leaf_id = builder.create_leaf_node(modality.modality_id)
    return builder.build(builder.create_operation_node(operation, [leaf_id], {}))


def _indexed(data, modality_id=0):
    modality = ModalityRandomDataGenerator().create1DModality(
        len(data), 4, ModalityType.TIMESERIES
    )
    modality.data = data
    modality.modality_id = modality_id
    return IndexedModality(modality, range(len(data)))


class _RecordingNodeCache:
    def __init__(self):
        self.entries = {}

    def get(self, node_id, extra=()):
        return self.entries.get((node_id, extra))

    def put(self, node_id, value, compute_seconds, extra=()):
        self.entries[(node_id, extra)] = value


class TestEvaluationSplit(unittest.TestCase):
    """A split lists the instances a fold is fitted on and the ones it is
    scored on. An instance in both parts would leak into the score, and an
    instance listed twice would count twice."""

    def test_split_rejects_repeated_or_shared_ids(self):
        malformed = {
            "repeated fit id": ((0, 0, 1), (2,)),
            "repeated evaluation id": ((0, 1), (2, 2)),
            "shared id": ((0, 1), (1, 2)),
        }
        for case, (fit_indices, evaluation_indices) in malformed.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    EvaluationSplit(fit_indices, evaluation_indices)


class TestSplitAwareFitting(unittest.TestCase):
    """A representation that needs fitting is fitted once per fold, on the
    fit part of that fold only. A supervised one also gets labels, and these
    have to be the labels of exactly those instances."""

    num_instances = 10

    def test_supervised_fit_gets_only_the_labels_of_its_fold(self):
        text = _text_modality(self.num_instances)
        task = TestTask("SupervisedFitTask", "Test1", self.num_instances)
        dag = _single_operation_dag(text, _SupervisedTfIdf)

        fits = []
        tfidf_fit = TfIdf.fit

        def recording_fit(op, inputs, labels=None):
            fits.append((inputs.instance_ids, labels))
            return tfidf_fit(op, inputs, labels=labels)

        with patch.object(_SupervisedTfIdf, "fit", recording_fit):
            SplitAwareDAGEvaluator(dag, text, task).evaluate()

        self.assertEqual(len(fits), len(task.evaluation_splits))
        for (fit_ids, labels), split in zip(fits, task.evaluation_splits):
            with self.subTest(split=split.name):
                self.assertEqual(fit_ids, split.fit_indices)
                np.testing.assert_array_equal(
                    labels, np.asarray(task.labels)[list(split.fit_indices)]
                )


class TestSplitAwareNodeCache(unittest.TestCase):
    """The node cache lets evaluations reuse intermediate results. A fitted
    result depends on the instances it was fitted on, so without share_fitted
    it is never put into the cache."""

    num_instances = 10

    def test_fitted_nodes_are_not_cached_by_default(self):
        text = _text_modality(self.num_instances)
        task = TestTask("NodeCacheTask", "Test1", self.num_instances)
        # the fitted node is not the root, since the root is never cached anyway
        builder = RepresentationDAGBuilder()
        tfidf_id = builder.create_operation_node(
            TfIdf, [builder.create_leaf_node(text.modality_id)], {}
        )
        dag = builder.build(
            builder.create_operation_node(
                MLPAveraging, [tfidf_id], MLPAveraging().get_current_parameters()
            )
        )

        fits = []
        tfidf_fit = TfIdf.fit

        def recording_fit(op, inputs, labels=None):
            fits.append(inputs.instance_ids)
            return tfidf_fit(op, inputs, labels=labels)

        node_cache = _RecordingNodeCache()
        with patch.object(TfIdf, "fit", recording_fit):
            SplitAwareDAGEvaluator(dag, text, task, node_cache=node_cache).evaluate()

        self.assertEqual(len(fits), len(task.evaluation_splits))
        self.assertEqual(node_cache.entries, {})


class TestLeafInput(unittest.TestCase):
    """A fusion DAG can get several inputs of the same modality, one per
    representation the unimodal search kept. Each leaf names the modality and
    the representation it reads."""

    def setUp(self):
        self.other = _indexed(np.zeros((3, 4)), modality_id=0)
        self.first = _indexed(np.zeros((3, 4)), modality_id=1)
        self.second = _indexed(np.zeros((3, 4)), modality_id=1)
        self.partition = (self.other, self.first, self.second)

    def _leaf_input(self, modality_id, representation_index):
        leaf = RepresentationNode("leaf", None, [], modality_id, representation_index)
        return SplitAwareDAGEvaluator._leaf_input(self.partition, leaf)

    def test_leaf_picks_its_modality_and_representation(self):
        for (modality_id, representation_index), expected in {
            (0, 0): self.other,
            (1, -1): self.first,
            (1, 1): self.second,
            ("1", 1): self.second,
        }.items():
            with self.subTest(
                modality_id=modality_id, representation_index=representation_index
            ):
                self.assertIs(
                    self._leaf_input(modality_id, representation_index), expected
                )

    def test_leaf_without_a_matching_input_raises(self):
        with self.assertRaises(KeyError):
            self._leaf_input(9, 0)
        with self.assertRaises(IndexError):
            self._leaf_input(1, 5)


class TestAlignPartitionShapes(unittest.TestCase):
    """A representation that pads its output pads it to the longest instance
    it sees. Applied to the fit, evaluation and test part one by one, the
    parts can end up with different lengths, which one model cannot take."""

    def test_shorter_partitions_are_zero_padded_to_the_longest(self):
        fit = _indexed(np.ones((3, 2, 4)))
        evaluation = _indexed([np.ones((3, 4))])
        test = _indexed(np.ones((2, 5, 4)))
        test_data = test.modality.data

        SplitAwareDAGEvaluator._align_partition_shapes((fit, evaluation, test))

        self.assertEqual(fit.modality.data.shape, (3, 5, 4))
        np.testing.assert_array_equal(fit.modality.data[:, :2], 1)
        np.testing.assert_array_equal(fit.modality.data[:, 2:], 0)
        self.assertIsInstance(evaluation.modality.data, list)
        self.assertEqual(evaluation.modality.data[0].shape, (5, 4))
        np.testing.assert_array_equal(evaluation.modality.data[0][3:], 0)
        self.assertIs(test.modality.data, test_data)

    def test_partitions_are_left_alone_when_padding_does_not_apply(self):
        for case, shapes in {
            "same shape": ((3, 2, 4), (2, 2, 4)),
            "different rank": ((3, 4), (2, 5, 4)),
        }.items():
            with self.subTest(case=case):
                partitions = tuple(_indexed(np.ones(shape)) for shape in shapes)
                before = [item.modality.data for item in partitions]
                SplitAwareDAGEvaluator._align_partition_shapes(partitions)
                for item, data in zip(partitions, before):
                    self.assertIs(item.modality.data, data)


class TestMaterialize(unittest.TestCase):
    """Split-aware evaluation picks instances by id, so every instance has to
    be in memory. A modality without data and without a loader to read it
    from cannot be evaluated."""

    def test_modality_without_data_or_loader_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "materialized data"):
            SplitAwareDAGEvaluator.materialize(Modality(ModalityType.TIMESERIES))


if __name__ == "__main__":
    unittest.main()
