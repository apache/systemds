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
from systemds.scuro.utils.static_variables import get_global_seed, get_train_seed
from systemds.scuro.models.model import Model
from systemds.scuro.models.fused_mlp import (
    candidate_slots,
    fit_folds,
    fused_training_enabled,
    supports_fused_training,
)
from systemds.scuro.models.training_stats import TrainingWorkAccumulator
import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from systemds.scuro.drsearch.evaluation_split import EvaluationSplit
from systemds.scuro.representations.representation import RepresentationStats

_GPU_CONTEXT_FLOOR_BYTES = (
    int(os.environ.get("SCURO_GPU_CONTEXT_FLOOR_MB", "512")) * 1024 * 1024
)


def _as_features(X):
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(len(X), -1)


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
        groups=None,
        allow_group_overlap=False,
        cv_seed=None,
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
        self.groups = None if groups is None else np.asarray(groups)
        self.allow_group_overlap = allow_group_overlap
        self.cv_seed = get_global_seed() if cv_seed is None else cv_seed
        self.measure_performance = measure_performance
        self.inference_time = []
        self.training_time = []
        self.last_run_timing = {}
        self.training_work = TrainingWorkAccumulator()
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
        if fused_training_enabled() and supports_fused_training(self.model):
            uses_gpu = False

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
        num_steps = getattr(self.model, "num_steps", None)
        if num_steps is not None:
            batch_size = max(1, int(getattr(self.model, "batch_size", 1) or 1))
            samples = max(1, int(num_steps)) * batch_size
            return float(max(1, self.kfold) * samples * max(1, feature_dim))
        passes = max(1, int(getattr(self.model, "epochs", 1) or 1))
        return float(
            max(1, self.kfold) * passes * max(1, n_fold_train) * max(1, feature_dim)
        )

    def validate_input_memory(self, data, gpu_memory_fraction: float = 0.9) -> None:
        if data is None or len(data) == 0:
            return

        if isinstance(data, np.ndarray):
            num_instances = int(data.shape[0]) if data.ndim else 1
            output_shape = tuple(data.shape[1:]) if data.ndim > 1 else (1,)
            dtype = data.dtype
        else:
            sample = np.asarray(data[0])
            num_instances = len(data)
            output_shape = tuple(sample.shape) if sample.ndim else (1,)
            dtype = sample.dtype

        estimate = self.estimate_peak_memory_bytes(
            RepresentationStats(
                num_instances,
                output_shape,
                output_shape_is_known=True,
                dtype=dtype,
            )
        )
        required = int(estimate.get("gpu_peak_bytes", 0))
        if required <= 0:
            return

        try:
            import torch

            if not torch.cuda.is_available():
                return
            device = getattr(self.model, "device", torch.device("cuda"))
            device_index = (
                device.index if getattr(device, "index", None) is not None else 0
            )
            free, total = torch.cuda.mem_get_info(device_index)
            cached = torch.cuda.memory_reserved(
                device_index
            ) - torch.cuda.memory_allocated(device_index)
        except (ImportError, RuntimeError):
            return

        available = int((free + max(cached, 0)) * gpu_memory_fraction)
        if required > available:
            feature_dim = int(np.prod(output_shape)) if output_shape else 0
            raise MemoryError(
                f"{self.model.name} estimated CUDA peak "
                f"{required / 1024**3:.2f} GiB for input_dim={feature_dim:,}, but "
                f"only {free / 1024**3:.2f} GiB is free "
                f"({total / 1024**3:.2f} GiB total). Rejecting this candidate "
                "before allocating CUDA tensors."
            )

    @staticmethod
    def _stratification_target(labels_array):
        if labels_array.ndim == 1:
            return labels_array
        return None

    def _fold_candidates(self, y, n_units):
        n_splits = max(2, min(self.kfold, int(n_units)))
        stratify_on = self._stratification_target(y)
        candidates = []
        if self.groups is not None:
            if stratify_on is not None:
                candidates.append(
                    (
                        StratifiedGroupKFold(
                            n_splits=n_splits, shuffle=True, random_state=self.cv_seed
                        ),
                        stratify_on,
                    )
                )
            candidates.append((GroupKFold(n_splits=n_splits), None))
        else:
            if stratify_on is not None:
                candidates.append(
                    (
                        StratifiedKFold(
                            n_splits=n_splits, shuffle=True, random_state=self.cv_seed
                        ),
                        stratify_on,
                    )
                )
            candidates.append(
                (
                    KFold(n_splits=n_splits, shuffle=True, random_state=self.cv_seed),
                    None,
                )
            )
        return candidates

    def _check_train_test_groups(self):
        if self.groups is None or not len(self.test_indices or []):
            return
        train_groups = set(self.groups[np.asarray(self.train_indices)].tolist())
        test_groups = set(self.groups[np.asarray(self.test_indices)].tolist())
        shared = train_groups & test_groups
        if shared and not self.allow_group_overlap:
            raise ValueError(
                f"{self.name}: train and test share {len(shared)} group(s) "
                f"(e.g. {sorted(map(str, shared))[:5]}). Use a group-disjoint "
                "outer split, or pass allow_group_overlap=True to declare a "
                "group-dependent protocol explicitly."
            )
        self.train_test_group_overlap = sorted(map(str, shared))

    def _create_cv_splits(self):
        self._check_train_test_groups()

        train_indices_array = np.asarray(self.train_indices)
        y = np.asarray([self.labels[i] for i in self.train_indices])
        fold_groups = None if self.groups is None else self.groups[train_indices_array]

        n_units = (
            len(train_indices_array)
            if fold_groups is None
            else len(set(fold_groups.tolist()))
        )
        folds = None
        reason = None
        for splitter, stratify_on in self._fold_candidates(y, n_units):
            try:
                folds = list(
                    splitter.split(train_indices_array, stratify_on, groups=fold_groups)
                )
            except ValueError as error:
                reason = f"{type(splitter).__name__}: {error}"
                continue
            self.cv_stratified = stratify_on is not None
            self.cv_fallback_reason = reason
            break
        if folds is None:
            raise ValueError(
                f"{self.name}: cannot build {self.kfold} folds over "
                f"{len(train_indices_array)} training instances "
                f"({n_units} unit(s)): {reason}"
            )

        self.kfold = len(folds)

        self.cv_grouped = self.groups is not None
        self.cv_train_indices = []
        self.cv_val_indices = []

        for fold_idx, (fit_positions, val_positions) in enumerate(folds):
            fold_train_indices = train_indices_array[fit_positions].tolist()
            fold_val_indices = train_indices_array[val_positions].tolist()

            self.cv_train_indices.append(fold_train_indices)
            self.cv_val_indices.append(fold_val_indices)

            overlap = set(fold_train_indices) & set(fold_val_indices)
            if overlap:
                raise ValueError(
                    f"Fold {fold_idx}: Overlap detected between train and val indices: {overlap}"
                )

            if fold_groups is not None:
                shared_groups = set(fold_groups[fit_positions].tolist()) & set(
                    fold_groups[val_positions].tolist()
                )
                if shared_groups:
                    raise ValueError(
                        f"Fold {fold_idx}: groups appear in both the fit and the "
                        f"validation part: {sorted(map(str, shared_groups))[:5]}"
                    )

        self.evaluation_splits = tuple(
            EvaluationSplit(train_indices, val_indices, name=f"fold_{fold_idx}")
            for fold_idx, (train_indices, val_indices) in enumerate(
                zip(self.cv_train_indices, self.cv_val_indices)
            )
        )

    def split_report(self):
        report = {
            "task": self.name,
            "n_train": len(self.train_indices),
            "n_test": 0 if self.test_indices is None else len(self.test_indices),
            "kfold": self.kfold,
            "grouped": bool(getattr(self, "cv_grouped", False)),
            "stratified": bool(getattr(self, "cv_stratified", False)),
            "cv_seed": self.cv_seed,
        }
        if self.groups is not None:
            report["n_groups_train"] = len(
                set(self.groups[np.asarray(self.train_indices)].tolist())
            )
            if len(self.test_indices or []):
                report["n_groups_test"] = len(
                    set(self.groups[np.asarray(self.test_indices)].tolist())
                )
            report["train_test_group_overlap"] = len(
                getattr(self, "train_test_group_overlap", [])
            )
        if getattr(self, "cv_fallback_reason", None):
            report["stratification_fallback"] = self.cv_fallback_reason
        return report

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
        self.validate_input_memory(data)
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
            **self.training_work.as_dict(),
        }
        return [
            self.train_scores.compute_averages(),
            self.val_scores.compute_averages(),
            self.test_scores.compute_averages(),
        ]

    def run_many(self, datas, device):
        model = self.create_model()
        rows = list(
            dict.fromkeys(list(self.train_indices) + list(self.test_indices or []))
        )
        local = {row: i for i, row in enumerate(rows)}
        splits = [
            ([local[i] for i in fit], [local[i] for i in val])
            for fit, val in zip(self.cv_train_indices, self.cv_val_indices)
        ]
        test = [local[i] for i in self.test_indices or []]
        labels = np.asarray(self._gather_by_indices(self.labels, rows))
        prepare = getattr(model, "prepare_features", None) or _as_features

        results, by_width = [None] * len(datas), {}
        for c, data in enumerate(datas):
            try:
                X = prepare(self._gather_by_indices(data, rows))
                if not np.isfinite(X).all():
                    raise ValueError("non-finite input features")
                by_width.setdefault(X.shape[1], []).append((c, X))
            except Exception as e:
                results[c] = e

        seed = getattr(model, "seed", None)
        seed = get_train_seed(get_global_seed() if seed is None else seed)
        outputs = {}
        start = time.perf_counter()
        for width, group in by_width.items():
            slots = candidate_slots(model, width, len(rows), self.kfold)
            for i in range(0, len(group), slots):
                chunk = group[i : i + slots]
                fitted = fit_folds(
                    model, [X for _, X in chunk], labels, splits, seed, device
                )
                outputs.update(zip([c for c, _ in chunk], fitted))
        per_fit_s = (time.perf_counter() - start) / max(1, len(outputs) * self.kfold)

        for c, out in outputs.items():
            if not np.isfinite(out).all():
                results[c] = ValueError("training diverged: non-finite model output")
                continue
            self._reset_params()
            for k, (fit, val) in enumerate(splits):
                for measure, idx in (
                    (self.train_scores, fit),
                    (self.val_scores, val),
                    (self.test_scores, test),
                ):
                    measure.add_scores(
                        model.scores_from_output(out[k][idx], labels[idx])[0]
                    )
            self.last_run_timing = {
                "train_time_per_fold_s": [per_fit_s] * self.kfold,
                "train_time_mean_s": per_fit_s,
                "n_test_instances": len(test),
                "fused_batch_candidates": len(outputs),
            }
            results[c] = [
                self.train_scores.compute_averages(),
                self.val_scores.compute_averages(),
                self.test_scores.compute_averages(),
            ]
        return results

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
        self.training_work = TrainingWorkAccumulator()
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
        # No-op for the non-neural models that share this machinery.
        self.training_work.add(model)

    @staticmethod
    def _gather_by_indices(values, indices):
        return [values[i] for i in indices]
