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

import numpy as np

from systemds.scuro.drsearch.task import Task
from tests.scuro.data_generator import TestSVM


def _task(labels, num_train, **kwargs):
    return Task(
        "CrossValidationTask",
        TestSVM("Test1"),
        labels,
        list(range(num_train)),
        list(range(num_train, len(labels))),
        kfold=4,
        **kwargs,
    )


class TestGroupedCrossValidation(unittest.TestCase):
    """Instances of one group, e.g. the recordings of one subject, are not
    independent. If a group ends up on both sides of a fold, or of the train
    and test split, the model is scored on someone it was trained on."""

    # 20 instances, two per group, both labels in every group
    labels = np.array([0, 1] * 10)
    groups = np.repeat(np.arange(10), 2)

    def test_folds_keep_each_group_on_one_side(self):
        task = _task(self.labels, 16, groups=self.groups)

        self.assertEqual(len(task.evaluation_splits), 4)
        for split in task.evaluation_splits:
            with self.subTest(split=split.name):
                fit_groups = set(self.groups[list(split.fit_indices)])
                evaluation_groups = set(self.groups[list(split.evaluation_indices)])
                self.assertEqual(fit_groups & evaluation_groups, set())

    def test_train_and_test_must_not_share_a_group(self):
        # instance 14 is trained on and 15 is tested on, both are in group 7
        with self.assertRaisesRegex(ValueError, "share 1 group"):
            _task(self.labels, 15, groups=self.groups)

    def test_allowed_overlap_is_recorded(self):
        task = _task(self.labels, 15, groups=self.groups, allow_group_overlap=True)
        self.assertEqual(task.train_test_group_overlap, ["7"])


class TestStratifiedCrossValidation(unittest.TestCase):
    """Folds keep the label balance of the training set when the labels allow
    it. Multi-label targets cannot be stratified, so they are split without
    stratification right away instead of after a failed attempt."""

    def test_only_single_label_targets_are_stratified(self):
        for case, (labels, stratified) in {
            "single label": (np.array([0, 1] * 10), True),
            "multi label": (np.array([[0, 1], [1, 0]] * 10), False),
        }.items():
            with self.subTest(case=case):
                task = _task(labels, 16)
                self.assertEqual(task.cv_stratified, stratified)
                self.assertIsNone(task.cv_fallback_reason)


if __name__ == "__main__":
    unittest.main()
