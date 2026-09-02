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
import os
import time
from typing import List
from systemds.scuro.models.model import Model
import numpy as np
from sklearn.model_selection import train_test_split
from systemds.scuro.drsearch.evaluation_split import EvaluationSplit
from systemds.scuro.representations.representation import RepresentationStats

_GPU_CONTEXT_FLOOR_BYTES = (
    int(os.environ.get("SCURO_GPU_CONTEXT_FLOOR_MB", "512")) * 1024 * 1024
)


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

    def fold_scores(self):
        return {
            metric: [float(v) for v in values] for metric, values in self.scores.items()
        }

    def score_stds(self):
        """Per-metric standard deviation across folds (0.0 for a single fold)."""
        return {
            metric: (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            for metric, values in self.scores.items()
        }


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
        self.last_run_timing = {}
        self.expected_dim = 1
        self.performance_measures = performance_measures
        self.train_scores = PerformanceMeasure("train", performance_measures)
        self.val_scores = PerformanceMeasure("val", performance_measures)
        self.test_scores = PerformanceMeasure("test", performance_measures)
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

        uses_gpu = getattr(self.model, "uses_gpu", None)
        if uses_gpu is None:
            uses_gpu = model_peak_memory_gpu > 0

        gpu_peak_bytes = int(model_peak_memory_gpu * 1.4) if uses_gpu else 0
        if uses_gpu and gpu_peak_bytes <= 0:
            gpu_peak_bytes = _GPU_CONTEXT_FLOOR_BYTES

        return {
            "cpu_peak_bytes": int(
                model_peak_memory_cpu * 1.5 + scheduler_side_cpu_bytes * 2
            ),
            "gpu_peak_bytes": gpu_peak_bytes,
        }

    def estimate_relative_cost(self, input_stats) -> float:
        feature_dim = (
            int(np.prod(input_stats.output_shape)) if input_stats.output_shape else 0
        )
        fold_sizes = [len(fold) for fold in self.cv_train_indices]
        n_fold_train = max(fold_sizes, default=len(self.train_indices or []))
        passes = max(1, int(getattr(self.model, "epochs", 1) or 1))
        return float(
            max(1, self.kfold) * passes * max(1, n_fold_train) * max(1, feature_dim)
        )

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

        self.evaluation_splits = tuple(
            EvaluationSplit(train_indices, val_indices, name=f"fold_{fold_idx}")
            for fold_idx, (train_indices, val_indices) in enumerate(
                zip(self.cv_train_indices, self.cv_val_indices)
            )
        )

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
        if hasattr(model, "clean_up"):
            model.clean_up()
            del model

        self.last_run_timing = {
            "train_time_per_fold_s": list(self.training_time),
            "test_inference_time_per_fold_s": list(self.inference_time),
            "train_time_mean_s": (
                float(np.mean(self.training_time)) if self.training_time else 0.0
            ),
            "test_inference_time_mean_s": (
                float(np.mean(self.inference_time)) if self.inference_time else 0.0
            ),
            "n_test_instances": len(self.test_indices) if self.test_indices else 0,
        }
        return [
            self.train_scores.compute_averages(),
            self.val_scores.compute_averages(),
            self.test_scores.compute_averages(),
        ]

    def fit_once_and_time_inference(
        self, data, latency_repeats: int = 200, latency_warmup: int = 20
    ):
        model = self.create_model()

        train_X = self._gather_by_indices(data, self.train_indices)
        train_y = self._gather_by_indices(self.labels, self.train_indices)
        test_X = self._gather_by_indices(data, self.test_indices)
        test_y = self._gather_by_indices(self.labels, self.test_indices)

        t0 = time.perf_counter()
        model.fit(train_X, train_y, test_X, test_y)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        test_score = model.test(np.asarray(test_X), test_y)
        batch_inference_time = time.perf_counter() - t0

        latency = self._measure_single_sample_latency(
            model, test_X, test_y, latency_repeats, latency_warmup
        )

        scores = PerformanceMeasure("test_single_fit", self.performance_measures)
        scores.add_scores(test_score[0])
        scores.compute_averages()

        if hasattr(model, "clean_up"):
            model.clean_up()

        n_test = len(test_X)
        return {
            "single_fit_train_time_s": fit_time,
            "test_batch_inference_time_s": batch_inference_time,
            "test_inference_time_per_instance_ms": (
                batch_inference_time / n_test * 1000.0 if n_test else 0.0
            ),
            "n_test_instances": n_test,
            "single_fit_test_scores": scores.average_scores,
            **latency,
        }

    @staticmethod
    def _prediction_callable(model):
        if hasattr(model, "predict"):
            return model.predict, "model.predict"
        clf = getattr(model, "clf", None)
        if clf is not None and hasattr(clf, "predict"):
            return clf.predict, "clf.predict"
        return None, None

    def _measure_single_sample_latency(
        self, model, test_X, test_y, repeats: int, warmup: int
    ):
        empty = {
            "inference_latency_ms_median": None,
            "inference_latency_ms_p95": None,
            "inference_latency_samples": 0,
            "inference_latency_source": None,
        }
        if repeats <= 0 or not len(test_X):
            return empty

        predict, source = self._prediction_callable(model)
        if predict is None:
            return empty

        timings = []
        for i in range(warmup + repeats):
            sample_X = np.asarray([test_X[i % len(test_X)]])
            t0 = time.perf_counter()
            try:
                predict(sample_X)
            except Exception:
                return empty
            elapsed = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                timings.append(elapsed)

        return {
            "inference_latency_ms_median": float(np.median(timings)),
            "inference_latency_ms_p95": float(np.percentile(timings, 95)),
            "inference_latency_samples": len(timings),
            "inference_latency_source": source,
        }

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
