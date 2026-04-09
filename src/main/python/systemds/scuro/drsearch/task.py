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
import copy
import time
from typing import List
from systemds.scuro.models.model import Model
import numpy as np
from sklearn.model_selection import train_test_split
from systemds.scuro.representations.representation import RepresentationStats


class PerformanceMeasure:
    def __init__(self, name, metrics, higher_is_better=True):
        self.average_scores = None
        self.name = name
        self.metrics = metrics
        self.higher_is_better = higher_is_better
        self.scores = {}

        if isinstance(metrics, list):
            for metric in metrics:
                self.scores[metric] = []
        else:
            self.scores[metrics] = []

    def add_scores(self, scores):
        if isinstance(self.metrics, list):
            for metric in self.metrics:
                self.scores[metric].append(scores[metric])
        else:
            self.scores[self.metrics].append(scores[self.metrics])

    def compute_averages(self):
        self.average_scores = {}
        if isinstance(self.metrics, list):
            for metric in self.metrics:
                self.average_scores[metric] = np.mean(self.scores[metric])
        else:
            self.average_scores[self.metrics] = np.mean(self.scores[self.metrics])
        return self


class Task:
    def __init__(
        self,
        name: str,
        model: Model,
        labels,
        train_indices: List,
        val_indices: List,
        kfold=5,
        measure_performance=True,
        performance_measures=["accuracy"],
        fusion_train_split=0.8,
    ):
        """
        Parent class for the prediction task that is performed on top of the aligned representation
        :param name: Name of the task
        :param model: ML-Model to run
        :param labels: Labels used for prediction
        :param train_indices: Indices to extract training data
        :param val_indices: Indices to extract validation data
        :param kfold: Number of crossvalidation runs

        """
        self.name = name
        self.model = model
        self.labels = labels
        self.train_indices = train_indices
        self.test_indices = val_indices
        self.kfold = kfold
        self.measure_performance = measure_performance
        self.inference_time = []
        self.training_time = []
        self.expected_dim = 1
        self.performance_measures = performance_measures
        self.train_scores = PerformanceMeasure("train", performance_measures)
        self.val_scores = PerformanceMeasure("val", performance_measures)
        self.test_scores = PerformanceMeasure("test", performance_measures)
        self.fusion_train_indices = None
        self._create_cv_splits()

    def get_output_stats(self, input_stats):
        # TODO: Implement a default estimate of the output stats for the task
        return RepresentationStats(0, (0,))

    def estimate_peak_memory_bytes(self, input_stats):
        labels_array = np.asarray(self.labels)
        n_instances = int(input_stats.num_instances)
        feature_dim = (
            int(np.prod(input_stats.output_shape)) if input_stats.output_shape else 0
        )
        feature_dtype_bytes = 4

        n_train = len(self.train_indices) if self.train_indices is not None else 0
        n_test = len(self.test_indices) if self.test_indices is not None else 0
        n_labels = int(labels_array.shape[1]) if labels_array.ndim > 1 else 1

        max_fold_train = max((len(fold) for fold in self.cv_train_indices), default=0)
        max_fold_val = max((len(fold) for fold in self.cv_val_indices), default=0)

        base_input_bytes = n_instances * feature_dim * feature_dtype_bytes
        test_slice_bytes = n_test * feature_dim * feature_dtype_bytes
        fold_slice_bytes = (
            (max_fold_train + max_fold_val) * feature_dim * feature_dtype_bytes
        )
        label_slice_bytes = (
            (max_fold_train + max_fold_val + n_test) * n_labels * feature_dtype_bytes
        )
        label_storage_bytes = int(labels_array.nbytes) * 2

        total_index_entries = (
            n_train
            + n_test
            + sum(len(fold) for fold in self.cv_train_indices)
            + sum(len(fold) for fold in self.cv_val_indices)
            + (
                len(self.fusion_train_indices)
                if self.fusion_train_indices is not None
                else 0
            )
        )
        index_bytes = total_index_entries * np.dtype(np.int64).itemsize

        scheduler_side_cpu_bytes = (
            base_input_bytes
            + test_slice_bytes * 2
            + fold_slice_bytes * 2
            + label_slice_bytes * 2
            + label_storage_bytes
            + index_bytes
            + (192 * 1024 * 1024)
        )

        model_peak_memory_cpu = 0.0
        model_peak_memory_gpu = 0.0
        if hasattr(self.model, "estimate_peak_memory_bytes"):
            try:
                model_peak_memory_cpu, model_peak_memory_gpu = (
                    self.model.estimate_peak_memory_bytes(
                        input_dim=feature_dim,
                        n_train_samples=(
                            max_fold_train if max_fold_train > 0 else n_train
                        ),
                        n_labels=n_labels,
                        n_val_samples=max_fold_val,
                        n_test_samples=n_test,
                    )
                )
            except TypeError:
                model_peak_memory_cpu, model_peak_memory_gpu = (
                    self.model.estimate_peak_memory_bytes(feature_dim, n_train)
                )

        return {
            "cpu_peak_bytes": int(
                model_peak_memory_cpu * 1.5 + scheduler_side_cpu_bytes * 2
            ),
            "gpu_peak_bytes": int(model_peak_memory_gpu) * 1.4,
        }

    def _create_cv_splits(self):
        train_labels = [self.labels[i] for i in self.train_indices]
        train_labels_array = np.array(train_labels)

        train_indices_array = np.array(self.train_indices)

        self.cv_train_indices = []
        self.cv_val_indices = []

        for fold_idx in range(self.kfold):
            fold_train_indices_array, fold_val_indices_array, _, _ = train_test_split(
                train_indices_array,
                train_labels_array,
                test_size=0.2,
                shuffle=True,
                random_state=11 + fold_idx,
            )

            fold_train_indices = fold_train_indices_array.tolist()
            fold_val_indices = fold_val_indices_array.tolist()

            self.cv_train_indices.append(fold_train_indices)
            self.cv_val_indices.append(fold_val_indices)

            overlap = set(fold_train_indices) & set(fold_val_indices)
            if overlap:
                raise ValueError(
                    f"Fold {fold_idx}: Overlap detected between train and val indices: {overlap}"
                )

        all_val_indices = set()
        for val_indices in self.cv_val_indices:
            all_val_indices.update(val_indices)

        self.fusion_train_indices = [
            idx for idx in self.train_indices if idx not in all_val_indices
        ]

    def create_model(self):
        """
        Return a fresh, unfitted model instance.
        """
        if self.model is None:
            return None

        return copy.deepcopy(self.model)

    def get_train_test_split(self, data):
        X_train = [data[i] for i in self.train_indices]
        y_train = [self.labels[i] for i in self.train_indices]
        if self.test_indices is None:
            X_test = None
            y_test = None
        else:
            X_test = [data[i] for i in self.test_indices]
            y_test = [self.labels[i] for i in self.test_indices]

        return X_train, y_train, X_test, y_test

    def run(self, data):
        """
        The run method needs to be implemented by every task class
         It handles the training and validation procedures for the specific task
         :param data: The aligned data used in the prediction process
         :return: the validation accuracy
        """
        self._reset_params()
        model = self.create_model()

        test_X = self._gather_by_indices(data, self.test_indices)
        test_y = self._gather_by_indices(self.labels, self.test_indices)

        for fold_idx in range(self.kfold):
            fold_train_indices = self.cv_train_indices[fold_idx]
            fold_val_indices = self.cv_val_indices[fold_idx]

            train_X = self._gather_by_indices(data, fold_train_indices)
            train_y = self._gather_by_indices(self.labels, fold_train_indices)
            val_X = self._gather_by_indices(data, fold_val_indices)
            val_y = self._gather_by_indices(self.labels, fold_val_indices)

            self._run_fold(model, train_X, train_y, val_X, val_y, test_X, test_y)

        return [
            self.train_scores.compute_averages(),
            self.val_scores.compute_averages(),
            self.test_scores.compute_averages(),
        ]

    def _reset_params(self):
        self.inference_time = []
        self.training_time = []
        self.train_scores = PerformanceMeasure("train", self.performance_measures)
        self.val_scores = PerformanceMeasure("val", self.performance_measures)
        self.test_scores = PerformanceMeasure("test", self.performance_measures)

    def _run_fold(self, model, train_X, train_y, val_X, val_y, test_X, test_y):
        train_start = time.time()
        train_score = model.fit(train_X, train_y, val_X, val_y)
        train_end = time.time()
        self.training_time.append(train_end - train_start)
        self.train_scores.add_scores(train_score[0])
        val_score = model.test(val_X, val_y)
        test_start = time.time()
        test_score = model.test(np.asarray(test_X), test_y)
        test_end = time.time()
        self.inference_time.append(test_end - test_start)
        self.val_scores.add_scores(val_score[0])
        self.test_scores.add_scores(test_score[0])

    @staticmethod
    def _gather_by_indices(values, indices):
        return [values[i] for i in indices]
