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

import inspect
import math
import os
from collections import defaultdict

import numpy as np

from systemds.scuro.modality.type import DataLayout, ModalityType

from systemds.scuro.drsearch.operator_registry import register_context_operator
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.context import Context
from systemds.scuro.representations.representation import (
    CONTAINER_ARRAY,
    CONTAINER_LIST,
    NDARRAY_OBJECT_OVERHEAD_BYTES,
    Representation,
    RepresentationStats,
    stats_itemsize,
)
from systemds.scuro.representations.utils import dense_instance_batch

_ACCEPTS_AXIS_CACHE = {}


def _accepts_axis(compute_feature):
    func = getattr(compute_feature, "__func__", compute_feature)
    accepts = _ACCEPTS_AXIS_CACHE.get(func)
    if accepts is None:
        accepts = "axis" in inspect.signature(compute_feature).parameters
        _ACCEPTS_AXIS_CACHE[func] = accepts
    return accepts


_WINDOW_FEATURE_BATCH_SIZE = max(
    1, int(os.environ.get("SCURO_WINDOW_FEATURE_BATCH_SIZE", "64"))
)


def _compute_window_features(aggregation_function, windows):
    """Compute independent windows in bounded vectorized batches when supported."""
    windows = [np.asarray(window) for window in windows]
    compute_batched = getattr(aggregation_function, "compute_features_batched", None)
    if not callable(compute_batched):
        return [aggregation_function.compute_feature(window) for window in windows]

    results = [None] * len(windows)
    grouped = defaultdict(list)
    for index, window in enumerate(windows):
        grouped[(window.shape, window.dtype.str)].append((index, window))

    for group in grouped.values():
        for start in range(0, len(group), _WINDOW_FEATURE_BATCH_SIZE):
            chunk = group[start : start + _WINDOW_FEATURE_BATCH_SIZE]
            batch = np.stack([window for _, window in chunk])
            batch_result = np.asarray(compute_batched(batch))
            if batch_result.ndim == 0 or batch_result.shape[0] != len(chunk):
                raise ValueError(
                    f"{aggregation_function.name}.compute_features_batched() must "
                    "preserve the leading batch dimension"
                )
            for row, (index, _) in enumerate(chunk):
                results[index] = batch_result[row]
    return results


def nested_aggregation_param_names(agg_cls):
    if not inspect.isclass(agg_cls):
        return set()
    if agg_cls is Aggregation or agg_cls.__name__ == "Aggregation":
        return {"aggregation_function", "pad_modality"}
    try:
        return set(agg_cls().parameters.keys())
    except Exception:
        return set()


def instantiate_nested_aggregation(agg_cls, nested):
    if not inspect.isclass(agg_cls):
        return agg_cls
    if not nested:
        return agg_cls()

    if agg_cls is Aggregation or agg_cls.__name__ == "Aggregation":
        return Aggregation(params=nested)

    allowed = nested_aggregation_param_names(agg_cls)
    filtered = {key: value for key, value in nested.items() if key in allowed}
    if not filtered:
        return agg_cls()

    init_params = inspect.signature(agg_cls.__init__).parameters
    if "params" in init_params:
        return agg_cls(params=filtered)
    return agg_cls(**filtered)


def _pad_stack(arrays):
    arrays = [np.asarray(a) for a in arrays]
    if len({a.shape for a in arrays}) == 1:
        return np.stack(arrays)

    ndim = max(a.ndim for a in arrays)
    arrays = [a.reshape((1,) * (ndim - a.ndim) + a.shape) for a in arrays]
    target_shape = tuple(max(a.shape[d] for a in arrays) for d in range(ndim))
    stacked = np.zeros((len(arrays), *target_shape), dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        slices = tuple(slice(0, s) for s in a.shape)
        stacked[(i, *slices)] = a
    return stacked


def _append_tail_row(full_result, tail_result):
    full_result = np.asarray(full_result)
    tail_result = np.asarray(tail_result)
    target_shape = full_result.shape[1:]
    if tail_result.shape == target_shape:
        tail_row = tail_result
    else:
        tail_row = np.zeros(target_shape, dtype=full_result.dtype)
        slices = tuple(
            slice(0, min(d, s)) for d, s in zip(target_shape, tail_result.shape)
        )
        tail_row[slices] = tail_result[slices]
    return np.concatenate([full_result, tail_row[None, ...]])


def _append_tail_rows(full_result, tail_result):
    full_result = np.asarray(full_result)
    tail_result = np.asarray(tail_result)
    target_shape = full_result.shape[2:]
    if tail_result.shape[1:] == target_shape:
        tail_rows = tail_result
    else:
        tail_rows = np.zeros(
            (full_result.shape[0], *target_shape), dtype=full_result.dtype
        )
        slices = tuple(
            slice(0, min(d, s)) for d, s in zip(target_shape, tail_result.shape[1:])
        )
        tail_rows[(slice(None), *slices)] = tail_result[(slice(None), *slices)]
    return np.concatenate([full_result, tail_rows[:, None, ...]], axis=1)


def resolve_aggregation_function(aggregation_function, params):
    if params is None:
        return aggregation_function
    if isinstance(params.get("aggregation_function"), (Aggregation, Representation)):
        return params["aggregation_function"]

    nested_agg = {
        key[len("aggregation_function_") :]: value
        for key, value in params.items()
        if key.startswith("aggregation_function_")
    }
    agg_value = params.get("aggregation_function")
    if nested_agg and inspect.isclass(agg_value):
        return instantiate_nested_aggregation(agg_value, nested_agg)
    if inspect.isclass(agg_value):
        return agg_value()
    return params.get("aggregation_function", aggregation_function)


class Window(Context):
    granularity_parameter = None
    granularity_kind = None  # "length" | "count"

    def __init__(self, name, aggregation_function):
        self.aggregation_function = aggregation_function
        parameters = {}
        if isinstance(self.aggregation_function, Aggregation) or isinstance(
            self.aggregation_function, Representation
        ):
            parameters["aggregation_function"] = self.aggregation_function.__class__
        else:
            raise ValueError(
                f"Invalid aggregation function type: {type(self.aggregation_function)}"
            )

        super().__init__(name, parameters)
        self.data_type = np.float32

    def get_current_parameters(self):
        current_params = {}
        if not self.parameters:
            return current_params

        for parameter in list(self.parameters.keys()):
            if parameter == "aggregation_function":
                if "aggregation_function" in self.parameters:
                    current_params["aggregation_function"] = self.parameters[
                        "aggregation_function"
                    ]
                    if isinstance(self.aggregation_function, Aggregation) or isinstance(
                        self.aggregation_function, Representation
                    ):
                        for (
                            key,
                            value,
                        ) in self.aggregation_function.get_current_parameters().items():
                            current_params[f"aggregation_function_{key}"] = value
            else:
                current_params[parameter] = getattr(self, parameter)
        return current_params

    @property
    def aggregation_function(self):
        return self._aggregation_function

    @aggregation_function.setter
    def aggregation_function(self, value):
        if isinstance(value, Representation):
            self._aggregation_function = value
        elif isinstance(value, Aggregation):
            self._aggregation_function = value
        else:
            self._aggregation_function = Aggregation(value)

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        return None

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        return None

    def assert_output_stats(self, windowed_data):
        if self.stats:
            assert (
                len(windowed_data) == self.stats.num_instances
            ), f"Output shape: {windowed_data.shape}, Expected shape: {self.stats.output_shape}"
            assert (
                windowed_data.shape[1] == self.stats.output_shape[0]
            ), f"Output shape: {windowed_data.shape}, Expected shape: {self.stats.output_shape}"
            feat_shape, bound = windowed_data.shape[2:], self.stats.output_shape[1:]
            assert len(feat_shape) == len(bound) and all(
                s <= b for s, b in zip(feat_shape, bound)
            ), f"Output shape: {windowed_data.shape}, Expected shape: {self.stats.output_shape}"

    @staticmethod
    def _shape_numel(shape):
        return int(np.prod(shape)) if len(shape) > 0 else 1

    @staticmethod
    def _rest_numel(shape):
        return int(np.prod(shape[1:])) if len(shape) > 1 else 1

    def _per_window_feature_shape(self, approx_window_size):
        windowed_input_stats = RepresentationStats(1, (approx_window_size,))
        feat_shape = self.aggregation_function.get_output_stats(
            windowed_input_stats
        ).output_shape
        return () if self._shape_numel(feat_shape) <= 1 else tuple(feat_shape)


@register_context_operator(
    [
        ModalityType.TIMESERIES,
        ModalityType.PHYSIOLOGICAL,
        ModalityType.AUDIO,
        ModalityType.EMBEDDING,
    ]
)
class WindowAggregation(Window):
    granularity_parameter = "window_size"
    granularity_kind = "length"

    def __init__(
        self,
        aggregation_function="mean",
        window_size=10,
        pad=True,
        params=None,
    ):
        window_size_set = False
        if isinstance(aggregation_function, Representation) and hasattr(
            aggregation_function, "window_size"
        ):
            window_size = aggregation_function.window_size
            window_size_set = True

        if params is not None:
            aggregation_function = resolve_aggregation_function(
                aggregation_function, params
            )
            if hasattr(aggregation_function, "window_size"):
                window_size = aggregation_function.window_size
                window_size_set = True

            window_size = params["window_size"] if not window_size_set else window_size
            pad = params.get("pad", True)
        super().__init__("WindowAggregation", aggregation_function)
        self.parameters["window_size"] = (4, 128)
        self.window_size = int(window_size)
        self.pad = pad

    def get_output_stats(self, input_stats: RepresentationStats) -> tuple:
        if not isinstance(self.aggregation_function, Aggregation):
            feat_shape = self._per_window_feature_shape(self.window_size)
            in_shape = (input_stats.output_shape[0], *feat_shape)
        else:
            in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 1:
            self.stats = RepresentationStats(
                input_stats.num_instances,
                (math.ceil(in_shape[0] / self.window_size),),
            )
        if len(in_shape) >= 2:
            self.stats = RepresentationStats(
                input_stats.num_instances,
                (math.ceil(in_shape[0] / self.window_size), *in_shape[1:]),
            )
        self.stats.output_shape_is_known = input_stats.output_shape_is_known
        return self.stats

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return 0

        out_seq_len = math.ceil(in_shape[0] / self.window_size)
        output_bytes = out_seq_len * self._rest_numel(in_shape)
        return input_stats.num_instances * output_bytes * stats_itemsize(input_stats)

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return {"cpu_peak_bytes": 0, "gpu_peak_bytes": 0}

        effective_seq_len = in_shape[0]
        in_numel = effective_seq_len * self._rest_numel(in_shape)
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        one_instance_bytes = in_numel * stats_itemsize(input_stats)
        input_bytes = one_instance_bytes * input_stats.num_instances

        list_bytes = output_bytes + input_stats.num_instances * (
            NDARRAY_OBJECT_OVERHEAD_BYTES
        )

        pad_bytes = 0
        if getattr(self, "pad", False):
            out_seq_len = math.ceil(in_shape[0] / self.window_size)
            padded_elems = (
                input_stats.num_instances * out_seq_len * self._rest_numel(in_shape)
            )
            pad_bytes = int(
                padded_elems * np.dtype(np.float64).itemsize
                + padded_elems * stats_itemsize(input_stats)
            )

        batch_bytes = (
            input_bytes
            if getattr(input_stats, "container", CONTAINER_ARRAY) == CONTAINER_LIST
            else 0
        )

        cpu_peak = int(
            (input_bytes + batch_bytes + list_bytes + pad_bytes) * 1.15
            + 16 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}

    def execute(self, modality):
        self.valid_windows = None
        batch = self._dense_batch(modality)
        if batch is not None:
            windowed_data = self.window_aggregate_single_level_batched(batch)
            if windowed_data is not None:
                if self.pad:
                    data_type = modality.metadata[0]["data_layout"]["type"]
                    if data_type != "str":
                        windowed_data = windowed_data.astype(data_type)
                else:
                    windowed_data = list(windowed_data)
                self.assert_output_stats(windowed_data)
                return windowed_data

        windowed_data = []
        original_lengths = []
        for instance in modality.data:
            new_length = math.ceil(len(instance) / self.window_size)
            if modality.get_data_layout() == DataLayout.SINGLE_LEVEL:
                instance = np.asarray(instance)
                instance.setflags(write=False)
                windowed_instance = self.window_aggregate_single_level(
                    instance, new_length
                )
            else:
                instance = np.array(instance)
                instance.setflags(write=False)
                windowed_instance = self.window_aggregate_nested_level(
                    instance, new_length
                )
            original_lengths.append(windowed_instance.shape[0])
            windowed_data.append(windowed_instance)

        if self.pad and not isinstance(windowed_data, np.ndarray):
            target_length = max(original_lengths)

            padded_features = np.zeros(
                (len(windowed_data), target_length, *windowed_data[0].shape[1:]),
                dtype=np.result_type(*windowed_data),
            )
            for i, features in enumerate(windowed_data):
                if padded_features.ndim == 3:
                    padded_features[i, : features.shape[0], :] = features
                else:
                    padded_features[i, : features.shape[0]] = features

            self.valid_windows = [(n, target_length) for n in original_lengths]
            windowed_data = padded_features
            data_type = modality.metadata[0]["data_layout"]["type"]
            if data_type != "str":
                windowed_data = windowed_data.astype(data_type, copy=False)

        self.assert_output_stats(windowed_data)
        return windowed_data

    def _dense_batch(self, modality):
        if modality.get_data_layout() != DataLayout.SINGLE_LEVEL:
            return None
        if not _accepts_axis(self.aggregation_function.compute_feature):
            return None

        batch = dense_instance_batch(modality.data)
        if batch is None:
            return None

        batch = batch.view()
        batch.setflags(write=False)
        return batch

    def window_aggregate_single_level_batched(self, data):
        num_instances, length = data.shape
        new_length = math.ceil(length / self.window_size)
        cut_length = (new_length - 1) * self.window_size
        tail = data[:, cut_length:]
        compute_feature = self.aggregation_function.compute_feature

        if new_length <= 1:
            if not tail.shape[1]:
                raise ValueError(
                    "Cannot window-aggregate an empty instance "
                    f"(window_size={self.window_size})."
                )
            if tail.shape[1] < self.window_size:
                pad_len = self.window_size - tail.shape[1]
                tail = np.pad(tail, ((0, 0), (0, pad_len)), mode="constant")
            result = np.asarray(compute_feature(tail, axis=1))
            if result.shape[0] != num_instances:
                return None
            return result[:, None, ...]

        full_batches = data[:, :cut_length].reshape(
            num_instances, new_length - 1, self.window_size
        )
        full_result = np.asarray(compute_feature(full_batches, axis=2))
        if full_result.shape[:2] != (num_instances, new_length - 1):
            return None

        if tail.shape[1]:
            tail_result = np.asarray(compute_feature(tail, axis=1))
            if tail_result.shape[0] != num_instances:
                return None
            full_result = _append_tail_rows(full_result, tail_result)

        return full_result

    def window_aggregate_single_level(self, instance, new_length):
        if isinstance(instance, str):
            return instance

        arr = np.asarray(instance)
        cut_length = (new_length - 1) * self.window_size
        tail = arr[cut_length:]
        takes_axis = _accepts_axis(self.aggregation_function.compute_feature)
        if new_length <= 1:
            if not tail.size:
                raise ValueError(
                    "Cannot window-aggregate an empty instance "
                    f"(window_size={self.window_size})."
                )
            if tail.shape[0] < self.window_size:
                pad_len = self.window_size - tail.shape[0]
                if tail.ndim == 1:
                    tail = np.pad(tail, (0, pad_len), mode="constant")
                else:
                    pad_width = [(0, 0)] * tail.ndim
                    pad_width[0] = (0, pad_len)
                    tail = np.pad(tail, pad_width=pad_width, mode="constant")
            if takes_axis:
                return np.array([self.aggregation_function.compute_feature(tail)])
            tail_result = self.aggregation_function.compute_feature(tail)
            return (
                tail_result[None, :]
                if tail_result.ndim > 0
                else np.array([tail_result])
            )
        full_batches = arr[:cut_length].reshape(
            new_length - 1, self.window_size, *arr.shape[1:]
        )

        if takes_axis:
            full_result = self.aggregation_function.compute_feature(
                full_batches, axis=1
            )
            if tail.size:
                tail_result = self.aggregation_function.compute_feature(tail)
                full_result = _append_tail_row(full_result, tail_result)
        else:
            windows = [full_batches[i] for i in range(full_batches.shape[0])]
            if tail.size:
                windows.append(tail)
            features = _compute_window_features(self.aggregation_function, windows)
            full_result = np.stack(features[: full_batches.shape[0]])
            if tail.size:
                full_result = _append_tail_row(full_result, features[-1])

        return full_result

    def window_aggregate_nested_level(self, instance, new_length):
        result = [[] for _ in range(0, new_length)]
        for i in range(0, new_length):
            result[i] = self.aggregation_function.compute_feature(
                instance[i * self.window_size : i * self.window_size + self.window_size]
            )

        return np.array(result)


@register_context_operator(
    [
        ModalityType.TIMESERIES,
        ModalityType.PHYSIOLOGICAL,
        ModalityType.AUDIO,
        ModalityType.EMBEDDING,
    ]
)
class StaticWindow(Window):
    granularity_parameter = "num_windows"
    granularity_kind = "count"

    def __init__(self, aggregation_function="mean", num_windows=100, params=None):
        if params is not None:
            aggregation_function = resolve_aggregation_function(
                aggregation_function, params
            )
            num_windows = params.get("num_windows", num_windows)

        super().__init__("StaticWindow", aggregation_function)
        self.parameters["num_windows"] = (min(5, num_windows), max(5, num_windows))
        self.num_windows = int(num_windows)

    def _feature_shape(self, in_shape):
        if isinstance(self.aggregation_function, Aggregation):
            return in_shape[1:]
        approx_window_size = (
            max(1, int(in_shape[0] / self.num_windows)) if in_shape else 1
        )
        return self._per_window_feature_shape(approx_window_size)

    def get_output_stats(self, input_stats: RepresentationStats) -> tuple:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        feat_shape = self._feature_shape(in_shape)
        self.stats = RepresentationStats(
            input_stats.num_instances, (self.num_windows, *feat_shape)
        )
        self.stats.output_shape_is_known = input_stats.output_shape_is_known

        return self.stats

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return 0

        out_shape = self.get_output_stats(input_stats).output_shape
        out_numel = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
        return input_stats.num_instances * out_numel * stats_itemsize(input_stats)

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return {"cpu_peak_bytes": 0, "gpu_peak_bytes": 0}
        effective_seq_len = in_shape[0]
        in_numel = effective_seq_len * self._rest_numel(in_shape)
        one_instance_bytes = in_numel * stats_itemsize(input_stats)
        input_bytes = one_instance_bytes * input_stats.num_instances
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        output_transient = output_bytes
        cpu_peak = int(
            (input_bytes + output_bytes + output_transient) * 1.12 + 12 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}

    def execute(self, modality):
        instance_windows = []
        for instance in modality.data:
            instance = np.asarray(instance)
            window_size = int(np.ceil(len(instance) / self.num_windows))
            padding_size = int(window_size * self.num_windows - len(instance))
            pad_width = [(0, 0)] * instance.ndim
            pad_width[0] = (0, padding_size)
            instance = np.pad(
                instance, pad_width=pad_width, mode="constant", constant_values=0
            )
            instance_windows.append(
                instance.reshape(self.num_windows, window_size, *instance.shape[1:])
            )

        if _accepts_axis(self.aggregation_function.compute_feature):
            windowed_data = [
                self.aggregation_function.compute_feature(windows, axis=1)
                for windows in instance_windows
            ]
        else:
            flat_windows = [
                window for windows in instance_windows for window in windows
            ]
            flat_features = _compute_window_features(
                self.aggregation_function, flat_windows
            )
            windowed_data = [
                np.stack(flat_features[start : start + self.num_windows])
                for start in range(0, len(flat_features), self.num_windows)
            ]

        return _pad_stack(windowed_data)


@register_context_operator(
    [
        ModalityType.TIMESERIES,
        ModalityType.PHYSIOLOGICAL,
        ModalityType.AUDIO,
        ModalityType.EMBEDDING,
    ]
)
class DynamicWindow(Window):
    granularity_parameter = "num_windows"
    granularity_kind = "count"

    def __init__(self, aggregation_function="mean", num_windows=100, params=None):
        if params is not None:
            aggregation_function = resolve_aggregation_function(
                aggregation_function, params
            )
            num_windows = params.get("num_windows", num_windows)
        super().__init__("DynamicWindow", aggregation_function)
        self.parameters["num_windows"] = (min(5, num_windows), max(5, num_windows))
        self.num_windows = int(num_windows)

    def _effective_num_windows(self, signal_length: int) -> int:
        if signal_length <= 0:
            return max(1, self.num_windows)
        return max(1, min(self.num_windows, int(signal_length)))

    def _window_sizes(self, signal_length: int) -> np.ndarray:
        num_windows = self._effective_num_windows(signal_length)
        length = max(int(signal_length), num_windows)
        weights = np.geomspace(4, 256, num=num_windows)
        weights = weights / np.sum(weights)

        sizes = 1 + (weights * (length - num_windows)).astype(int)
        sizes[-1] += length - int(sizes.sum())
        return sizes

    def _max_window_size(self, max_length: int) -> int:
        # Upper bound on every window of every instance with at most max_length
        # elements. Window i < q-1 holds more than w_i (L - q) elements, so the
        # last one (which also takes the rounding remainder) holds fewer than
        # q + w_last (L - q); and no window exceeds L - q + 1.
        num_windows = self._effective_num_windows(max_length)
        length = max(int(max_length), num_windows)
        weights = np.geomspace(4, 256, num=num_windows)
        last_share = weights[-1] / np.sum(weights)
        geometric_bound = int(
            np.ceil(num_windows + last_share * (length - num_windows))
        )
        return max(1, min(length - num_windows + 1, geometric_bound))

    def _feature_shape(self, in_shape):
        if isinstance(self.aggregation_function, Aggregation):
            return in_shape[1:]
        length = in_shape[0] if in_shape else 0
        # Window sizes grow geometrically and _pad_stack pads every window's
        # features to the largest one, so size them from an upper bound on the
        # largest window (about 4x the mean) over all lengths up to L, not the
        # mean: a shorter instance can have a longer last window.
        max_window_size = self._max_window_size(length) if in_shape else 1
        return self._per_window_feature_shape(max_window_size)

    def get_output_stats(self, input_stats: RepresentationStats) -> tuple:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        feat_shape = self._feature_shape(in_shape)
        num_windows = self._effective_num_windows(in_shape[0] if in_shape else 0)
        self.stats = RepresentationStats(
            input_stats.num_instances, (num_windows, *feat_shape)
        )
        self.stats.output_shape_is_known = input_stats.output_shape_is_known
        return self.stats

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return 0

        out_shape = self.get_output_stats(input_stats).output_shape
        out_numel = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
        return input_stats.num_instances * out_numel * stats_itemsize(input_stats)

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return {"cpu_peak_bytes": 0, "gpu_peak_bytes": 0}
        effective_seq_len = in_shape[0]
        in_numel = effective_seq_len * self._rest_numel(in_shape)
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        one_instance_bytes = in_numel * stats_itemsize(input_stats)
        cpu_peak = (
            output_bytes * 2
            + one_instance_bytes * input_stats.num_instances
            + 8 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}

    def execute(self, modality):
        all_windows = []
        windows_per_instance = []

        for instance in modality.data:
            instance = np.asarray(instance)
            indices = np.cumsum(self._window_sizes(len(instance)))
            start = 0
            count = 0
            for end in indices:
                window = instance[start:end]
                window.setflags(write=False)
                all_windows.append(window)
                start = end
                count += 1
            windows_per_instance.append(count)

        all_features = _compute_window_features(self.aggregation_function, all_windows)
        windowed_data = []
        start = 0
        for count in windows_per_instance:
            windowed_data.append(_pad_stack(all_features[start : start + count]))
            start += count

        windowed_data = _pad_stack(windowed_data)
        self.assert_output_stats(windowed_data)
        return windowed_data
