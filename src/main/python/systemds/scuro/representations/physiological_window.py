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
import math
import numpy as np

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_context_operator
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.window_aggregation import (
    Window,
    resolve_aggregation_function,
    _pad_stack,
)


def _estimate_windowed_output_stats(
    window_obj, input_stats, estimated_num_windows, representative_window_length
):
    in_shape = tuple(int(s) for s in input_stats.output_shape)
    if not isinstance(window_obj.aggregation_function, Aggregation):
        windowed_input_stats = RepresentationStats(
            input_stats.num_instances, (representative_window_length,)
        )
        feat_shape = window_obj.aggregation_function.get_output_stats(
            windowed_input_stats
        ).output_shape
    else:
        feat_shape = in_shape[1:]
    window_obj.stats = RepresentationStats(
        input_stats.num_instances,
        (estimated_num_windows, *feat_shape),
        output_shape_is_known=False,
    )
    return window_obj.stats


def _estimate_windowed_memory_bytes(window_obj, input_stats):
    out_shape = window_obj.get_output_stats(input_stats).output_shape
    out_numel = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
    return (
        input_stats.num_instances * out_numel * np.dtype(window_obj.data_type).itemsize
    )


def _estimate_windowed_peak_memory_bytes(window_obj, input_stats):
    in_shape = tuple(int(s) for s in input_stats.output_shape)
    in_numel = int(np.prod(in_shape)) if len(in_shape) > 0 else 1
    input_bytes = (
        input_stats.num_instances * in_numel * np.dtype(window_obj.data_type).itemsize
    )
    output_bytes = _estimate_windowed_memory_bytes(window_obj, input_stats)
    cpu_peak = int((input_bytes + output_bytes * 2) * 1.2 + 8 * 1024 * 1024)
    return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}


@register_context_operator([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
class AdaptiveWindow(Window):
    granularity_parameter = "base_window_size"
    granularity_kind = "length"

    def __init__(
        self,
        aggregation_function="mean",
        base_window_size=256,
        overlap=0.5,
        min_window_size=64,
        params=None,
    ):
        if params is not None:
            aggregation_function = resolve_aggregation_function(
                aggregation_function, params
            )
            base_window_size = params.get("base_window_size", base_window_size)
            overlap = params.get("overlap", overlap)
            min_window_size = params.get("min_window_size", min_window_size)
        super().__init__("AdaptiveWindow", aggregation_function)
        base_window_size = max(1, int(base_window_size))
        min_window_size = max(1, min(int(min_window_size), base_window_size))
        self.parameters.update(
            {
                "base_window_size": (min(4, base_window_size), base_window_size),
                "overlap": [0.25, 0.5, 0.75],
                "min_window_size": (min(2, min_window_size), min_window_size),
            }
        )
        self.base_window_size = base_window_size
        self.overlap = overlap
        self.min_window_size = min_window_size

    def _estimate_num_windows(self, signal_length):
        if signal_length <= 0:
            return 1
        step = max(1, int(self.base_window_size * (1 - self.overlap)))
        return max(1, math.ceil(signal_length / step))

    def get_output_stats(self, input_stats: RepresentationStats) -> RepresentationStats:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        signal_length = in_shape[0] if in_shape else 0
        return _estimate_windowed_output_stats(
            self,
            input_stats,
            self._estimate_num_windows(signal_length),
            self.base_window_size,
        )

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        return _estimate_windowed_memory_bytes(self, input_stats)

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        return _estimate_windowed_peak_memory_bytes(self, input_stats)

    def execute(self, modality):
        windowed_data = []

        for signal in modality.data:
            local_var = np.array(
                [
                    np.var(signal[i : i + self.min_window_size])
                    for i in range(
                        0, len(signal) - self.min_window_size, self.min_window_size
                    )
                ]
            )

            if local_var.size == 0:
                window_sizes = np.array([self.base_window_size])
            else:
                norm_var = (local_var - np.min(local_var)) / (
                    np.max(local_var) - np.min(local_var) + 1e-6
                )
                window_sizes = np.clip(
                    self.base_window_size * (1 - 0.5 * norm_var),
                    self.min_window_size,
                    self.base_window_size,
                ).astype(int)

            windows = []
            start = 0
            while start < len(signal):
                current_size = window_sizes[
                    min(len(window_sizes) - 1, start // self.min_window_size)
                ]
                end = min(start + current_size, len(signal))
                window = signal[start:end]
                if len(window) > 0:
                    windows.append(window)
                start += max(1, int(current_size * (1 - self.overlap)))

            processed_windows = _pad_stack(
                [self.aggregation_function.compute_feature(w) for w in windows]
            )
            windowed_data.append(processed_windows)

        return _pad_stack(windowed_data)


@register_context_operator([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
class PhysiologicalEventWindow(Window):
    granularity_parameter = "min_distance"
    granularity_kind = "length"

    def __init__(
        self,
        aggregation_function="mean",
        event_threshold=0.5,
        min_distance=100,
        params=None,
    ):
        if params is not None:
            aggregation_function = resolve_aggregation_function(
                aggregation_function, params
            )
            event_threshold = params.get("event_threshold", event_threshold)
            min_distance = params.get("min_distance", min_distance)
        super().__init__("PhysiologicalEventWindow", aggregation_function)
        min_distance = max(1, int(min_distance))
        self.parameters.update(
            {
                "event_threshold": [0.3, 0.5, 0.7],
                "min_distance": (min(4, min_distance), min_distance),
            }
        )
        self.event_threshold = event_threshold
        self.min_distance = min_distance

    def _estimate_num_windows(self, signal_length):
        if signal_length <= 0:
            return 1
        return max(1, math.ceil(signal_length / max(1, self.min_distance)))

    def get_output_stats(self, input_stats: RepresentationStats) -> RepresentationStats:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        signal_length = in_shape[0] if in_shape else 0
        return _estimate_windowed_output_stats(
            self,
            input_stats,
            self._estimate_num_windows(signal_length),
            self.min_distance,
        )

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        return _estimate_windowed_memory_bytes(self, input_stats)

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        return _estimate_windowed_peak_memory_bytes(self, input_stats)

    def execute(self, modality):
        windowed_data = []

        for signal in modality.data:
            normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

            peaks = []
            last_peak = -self.min_distance
            for i in range(1, len(normalized) - 1):
                if (
                    normalized[i] > self.event_threshold
                    and normalized[i] > normalized[i - 1]
                    and normalized[i] > normalized[i + 1]
                    and i - last_peak >= self.min_distance
                ):
                    peaks.append(i)
                    last_peak = i

            windows = [signal[peaks[i] : peaks[i + 1]] for i in range(len(peaks) - 1)]

            if not windows:
                windows = [
                    w
                    for w in np.array_split(
                        signal, max(1, len(signal) // self.min_distance)
                    )
                    if len(w) > 0
                ]

            processed_windows = _pad_stack(
                [self.aggregation_function.compute_feature(w) for w in windows]
            )
            windowed_data.append(processed_windows)

        return _pad_stack(windowed_data)
