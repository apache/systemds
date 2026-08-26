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
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_context_representation_operator,
)


def _as_float1d(signal):
    return np.asarray(signal, dtype=np.float64).reshape(-1)


def _successive_diffs(intervals):
    intervals = _as_float1d(intervals)
    if intervals.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(intervals)


def _segment_duration(intervals):
    intervals = _as_float1d(intervals)
    if intervals.size == 0:
        return 0.0
    return float(np.sum(intervals))


def _interpolate_tachogram(nn_intervals, fs=4.0):
    nn = _as_float1d(nn_intervals)
    if nn.size < 2:
        return nn
    times = np.cumsum(nn)
    times = times - times[0]
    duration = times[-1]
    if duration <= 0.0:
        return nn
    t_uniform = np.arange(0.0, duration, 1.0 / fs)
    if t_uniform.size < 2:
        return nn
    return np.interp(t_uniform, times, nn)


def _bandpower(signal, fs, f1, f2):
    x = _as_float1d(signal)
    if x.size < 4:
        return 0.0
    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(x)) ** 2
    mask = (freqs >= f1) & (freqs < f2)
    return float(np.sum(psd[mask]))


def _detect_ecg_r_peaks(signal, fs, min_rr_s=0.3):
    x = _as_float1d(signal)
    min_distance = max(1, int(min_rr_s * fs))

    if x.size <= min_distance:
        return np.array([], dtype=int)
    # 99th percentile instead of max to stay robust to outlier spikes.
    height = np.quantile(x, 0.99) / 2.0
    peaks, _ = find_peaks(x, distance=min_distance, height=height)
    return peaks


def _ecg_nn_intervals(signal, fs):
    peaks = _detect_ecg_r_peaks(signal, fs)
    if peaks.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(peaks) / float(fs)


def _count_matches_vectorized(x, template_len, r):
    n = x.size
    n_templates = n - template_len
    if n_templates < 2:
        return 0
    templates = np.lib.stride_tricks.sliding_window_view(x, template_len)[:n_templates]
    dists = pdist(templates, metric="chebyshev")
    return int(np.sum(dists <= r))


def _sample_entropy(intervals, m=2, r_factor=0.2):
    x = _as_float1d(intervals)
    n = x.size
    if n <= m + 1:
        return 0.0
    r = r_factor * np.std(x)
    if r <= 0.0:
        return 0.0

    a = _count_matches_vectorized(x, m + 1, r)
    b = _count_matches_vectorized(x, m, r)
    if b == 0 or a == 0:
        return 0.0
    return float(-np.log(a / b))


def _fluctuation_for_scale(y, scale, n_segments):
    segments = y[: n_segments * scale].reshape(n_segments, scale)
    t = np.arange(scale, dtype=np.float64)
    t_mean = t.mean()
    t_centered = t - t_mean
    denom = np.sum(t_centered**2)
    if denom == 0.0:
        return None
    seg_mean = segments.mean(axis=1, keepdims=True)
    slope = (segments * t_centered[None, :]).sum(axis=1, keepdims=True) / denom
    intercept = seg_mean - slope * t_mean
    trend = intercept + slope * t[None, :]
    rms = np.sqrt(np.mean((segments - trend) ** 2, axis=1))
    return float(rms.mean())


def _dfa_alpha(signal, min_box=4, max_box=None):
    x = _as_float1d(signal)
    n = x.size
    if n < min_box * 2:
        return 0.0
    y = np.cumsum(x - np.mean(x))
    if max_box is None:
        max_box = max(min_box + 1, n // 4)
    if max_box <= min_box:
        return 0.0

    scales = np.unique(
        np.logspace(np.log10(min_box), np.log10(max_box), num=10, dtype=int)
    )
    fluctuations = []
    for scale in scales:
        if scale < min_box:
            continue
        n_segments = n // scale
        if n_segments < 1:
            continue
        fluctuation = _fluctuation_for_scale(y, int(scale), n_segments)
        if fluctuation is not None:
            fluctuations.append((scale, fluctuation))

    fluctuations = [(s, f) for s, f in fluctuations if f > 0.0]
    if len(fluctuations) < 2:
        return 0.0
    scales, f_vals = zip(*fluctuations)
    alpha = np.polyfit(np.log(scales), np.log(f_vals), 1)[0]
    return float(alpha)


def _detect_scr_peaks(signal, fs, min_distance_s=1.0, prominence_factor=0.05):
    x = _as_float1d(signal)
    min_distance = max(1, int(min_distance_s * fs))

    if x.size <= min_distance:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    prominence = max(prominence_factor * (np.max(x) - np.min(x)), 1e-12)
    peaks, props = find_peaks(x, distance=min_distance, prominence=prominence)

    if peaks.size == 0:
        return peaks, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    left = np.maximum(0, peaks - min_distance)
    right = np.minimum(x.size - 1, peaks + min_distance)
    window = min_distance + 1
    offsets = np.arange(window)

    k_max_left = peaks - left
    idx_left = np.clip(peaks[:, None] - offsets[None, :], 0, x.size - 1)
    vals_left = x[idx_left]
    valid_left = offsets[None, :] <= k_max_left[:, None]

    baseline = np.min(np.where(valid_left, vals_left, np.inf), axis=1)
    amplitudes = x[peaks] - baseline
    half = baseline + 0.5 * amplitudes

    below_left = (vals_left <= half[:, None]) & valid_left
    has_below_left = below_left.any(axis=1)
    first_below_left = np.argmax(below_left, axis=1)
    li = np.maximum(
        left, peaks - np.where(has_below_left, first_below_left, k_max_left)
    )

    m_max = right - peaks
    idx_right = np.clip(peaks[:, None] + offsets[None, :], 0, x.size - 1)
    vals_right = x[idx_right]
    valid_right = offsets[None, :] <= m_max[:, None]

    below_right = (vals_right <= half[:, None]) & valid_right
    has_below_right = below_right.any(axis=1)
    first_below_right = np.argmax(below_right, axis=1)
    ri = np.minimum(right, peaks + np.where(has_below_right, first_below_right, m_max))

    durations = (ri - li) / fs

    return (
        peaks,
        amplitudes.astype(np.float64),
        durations.astype(np.float64),
    )


class PhysiologicalRepresentation(UnimodalRepresentation):
    def __init__(self, name, parameters=None, params=None):
        if params is None:
            params = {}
        super().__init__(name, ModalityType.EMBEDDING, parameters, False)

    def compute_feature(self, signal):
        raise NotImplementedError("Subclasses should implement this method.")

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        result = []
        for signal in modality.data:
            result.append(self.compute_feature(signal))

        maxlen = max(r.size for r in result)
        padded_result = [
            np.pad(r, (0, maxlen - r.size), mode="constant", constant_values=0.0)
            for r in result
        ]
        dtype = modality.metadata[0]["data_layout"]["type"]
        transformed_modality.data = np.vstack(np.asarray(padded_result)).astype(dtype)
        return transformed_modality

    def get_output_stats(self, input_stats):
        return RepresentationStats(
            input_stats.num_instances, (1,), input_stats.output_shape_is_known
        )

    @staticmethod
    def _num_elements(shape) -> int:
        n = 1
        for d in shape:
            n *= int(d)
        return n

    def estimate_output_memory_bytes(self, input_stats):
        out_stats = self.get_output_stats(input_stats)
        return (
            int(out_stats.num_instances)
            * self._num_elements(out_stats.output_shape)
            * np.dtype(np.float32).itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats):
        n_per_instance = self._num_elements(input_stats.output_shape)
        input_bytes = (
            int(input_stats.num_instances)
            * n_per_instance
            * np.dtype(np.float32).itemsize
        )
        transient_bytes = n_per_instance * np.dtype(np.float64).itemsize
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        cpu_peak = (
            int((input_bytes + 2 * transient_bytes + output_bytes) * 1.15)
            + 4 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class SDNN(PhysiologicalRepresentation):
    def __init__(self, fs=500.0, params=None):
        super().__init__("SDNN")
        if params is not None:
            fs = params.get("fs", fs)
        self.fs = fs

    def compute_feature(self, signal):
        nn = _ecg_nn_intervals(signal, self.fs)
        if nn.size < 2:
            return np.array(0.0)
        return np.array(np.std(nn, ddof=1))


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class RMSSD(PhysiologicalRepresentation):
    def __init__(self, fs=500.0, params=None):
        super().__init__("RMSSD")
        if params is not None:
            fs = params.get("fs", fs)
        self.fs = fs

    def compute_feature(self, signal):
        nn = _ecg_nn_intervals(signal, self.fs)
        diffs = _successive_diffs(nn)
        if diffs.size == 0:
            return np.array(0.0)
        return np.array(np.sqrt(np.mean(diffs**2)))


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class pNN(PhysiologicalRepresentation):
    def __init__(self, threshold_ms=50, fs=500.0, params=None):
        super().__init__("pNN", parameters={"threshold_ms": [20, 50]})
        if params is not None:
            threshold_ms = params.get("threshold_ms", threshold_ms)
            fs = params.get("fs", fs)
        self.threshold_ms = threshold_ms
        self.fs = fs

    def compute_feature(self, signal):
        nn = _ecg_nn_intervals(signal, self.fs)
        diffs = _successive_diffs(nn)
        if diffs.size == 0:
            return np.array(0.0)
        threshold = self.threshold_ms / 1000.0
        return np.array(100.0 * np.mean(np.abs(diffs) > threshold))


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class RRPerMinute(PhysiologicalRepresentation):
    def __init__(self, fs=500.0, params=None):
        super().__init__("RRPerMinute")
        if params is not None:
            fs = params.get("fs", fs)
        self.fs = fs

    def compute_feature(self, signal):
        nn = _ecg_nn_intervals(signal, self.fs)
        duration = _segment_duration(nn)
        if duration <= 0.0 or nn.size == 0:
            return np.array(0.0)
        return np.array(60.0 * nn.size / duration)


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class HRVBandPower(PhysiologicalRepresentation):
    def __init__(self, fs=4.0, f1=0.04, f2=0.15, signal_fs=500.0, params=None):
        super().__init__(
            "HRVBandPower",
            parameters={
                "fs": [2.0, 4.0],
                "f1": [0.003, 0.04, 0.15],
                "f2": [0.04, 0.15, 0.4],
            },
        )
        if params is not None:
            fs = params.get("fs", fs)
            f1 = params.get("f1", f1)
            f2 = params.get("f2", f2)
            signal_fs = params.get("signal_fs", signal_fs)
        self.fs = fs
        self.f1 = f1
        self.f2 = f2
        self.signal_fs = signal_fs

    def compute_feature(self, signal):
        nn = _ecg_nn_intervals(signal, self.signal_fs)
        tach = _interpolate_tachogram(nn, fs=self.fs)
        return np.array(_bandpower(tach, self.fs, self.f1, self.f2))


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class HRVVLF(HRVBandPower):
    def __init__(self, fs=4.0, signal_fs=500.0, params=None):
        super().__init__(fs=fs, f1=0.003, f2=0.04, signal_fs=signal_fs, params=params)
        self.name = "HRVVLF"
        self._parameters = {"fs": [2.0, 4.0]}


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class HRVLF(HRVBandPower):
    def __init__(self, fs=4.0, signal_fs=500.0, params=None):
        super().__init__(fs=fs, f1=0.04, f2=0.15, signal_fs=signal_fs, params=params)
        self.name = "HRVLF"
        self._parameters = {"fs": [2.0, 4.0]}


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class HRVHF(HRVBandPower):
    def __init__(self, fs=4.0, signal_fs=500.0, params=None):
        super().__init__(fs=fs, f1=0.15, f2=0.40, signal_fs=signal_fs, params=params)
        self.name = "HRVHF"
        self._parameters = {"fs": [2.0, 4.0]}


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class HRVLFHF(PhysiologicalRepresentation):
    def __init__(self, fs=4.0, signal_fs=500.0, params=None):
        super().__init__("HRVLFHF", parameters={"fs": [2.0, 4.0]})
        if params is not None:
            fs = params.get("fs", fs)
            signal_fs = params.get("signal_fs", signal_fs)
        self.fs = fs
        self.signal_fs = signal_fs

    def compute_feature(self, signal):
        lf = HRVLF(fs=self.fs, signal_fs=self.signal_fs).compute_feature(signal)[()]
        hf = HRVHF(fs=self.fs, signal_fs=self.signal_fs).compute_feature(signal)[()]
        if hf <= 0.0:
            return np.array(0.0)
        return np.array(lf / hf)


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class PoincareSD1(PhysiologicalRepresentation):
    def __init__(self, fs=500.0, params=None):
        super().__init__("PoincareSD1")
        if params is not None:
            fs = params.get("fs", fs)
        self.fs = fs

    def compute_feature(self, signal):
        nn = _ecg_nn_intervals(signal, self.fs)
        diffs = _successive_diffs(nn)
        if diffs.size < 2:
            return np.array(0.0)
        return np.array(np.std(diffs, ddof=1) / np.sqrt(2.0))


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class PoincareSD2(PhysiologicalRepresentation):
    def __init__(self, fs=500.0, params=None):
        super().__init__("PoincareSD2")
        if params is not None:
            fs = params.get("fs", fs)
        self.fs = fs

    def compute_feature(self, signal):
        nn = _ecg_nn_intervals(signal, self.fs)
        if nn.size < 3:
            return np.array(0.0)
        summed = nn[:-1] + nn[1:]
        return np.array(np.std(summed, ddof=1) / np.sqrt(2.0))


# @register_representation([ModalityType.PHYSIOLOGICAL])
# @register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
# class SampleEntropy(PhysiologicalRepresentation):
#     def __init__(self, m=2, r_factor=0.2, params=None):
#         super().__init__(
#             "SampleEntropy", parameters={"m": [2, 3], "r_factor": [0.15, 0.2, 0.25]}
#         )
#         if params is not None:
#             m = params.get("m", m)
#             r_factor = params.get("r_factor", r_factor)
#         self.m = m
#         self.r_factor = r_factor

#     def compute_feature(self, nn):
#         return np.array(_sample_entropy(nn, m=self.m, r_factor=self.r_factor))


# @register_representation([ModalityType.PHYSIOLOGICAL])
# @register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
# class DFAAlpha(PhysiologicalRepresentation):
#     def __init__(self, params=None):
#         super().__init__("DFAAlpha")

#     def compute_feature(self, nn):
#         return np.array(_dfa_alpha(nn))


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class SCLSlope(PhysiologicalRepresentation):
    def __init__(self, params=None):
        super().__init__("SCLSlope")

    def compute_feature(self, scl):
        x = _as_float1d(scl)
        if x.size < 2:
            return np.array(0.0)
        t = np.arange(x.size, dtype=np.float64)
        return np.array(np.polyfit(t, x, 1)[0])


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class SCLDynamicRange(PhysiologicalRepresentation):
    def __init__(self, params=None):
        super().__init__("SCLDynamicRange")

    def compute_feature(self, scl):
        x = _as_float1d(scl)
        if x.size == 0:
            return np.array(0.0)
        return np.array(np.max(x) - np.min(x))


class _SCRFeature(PhysiologicalRepresentation):
    def __init__(self, name, fs=4.0, parameters=None, params=None):
        params_dict = parameters or {"fs": [2.0, 4.0, 8.0]}
        super().__init__(name, parameters=params_dict, params=params)
        self.fs = fs

    def _scr_stats(self, scr):
        duration = _as_float1d(scr).size / self.fs
        peaks, amplitudes, durations = _detect_scr_peaks(scr, self.fs)
        return duration, peaks, amplitudes, durations


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class SCRPeaksPerMinute(_SCRFeature):
    def __init__(self, fs=4.0, params=None):
        super().__init__("SCRPeaksPerMinute", fs=fs, params=params)

    def compute_feature(self, scr):
        duration, peaks, _, _ = self._scr_stats(scr)
        if duration <= 0.0:
            return np.array(0.0)
        return np.array(60.0 * peaks.size / duration)


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class SCRAverageAmplitude(_SCRFeature):
    def __init__(self, fs=4.0, params=None):
        super().__init__("SCRAverageAmplitude", fs=fs, params=params)

    def compute_feature(self, scr):
        _, _, amplitudes, _ = self._scr_stats(scr)
        if amplitudes.size == 0:
            return np.array(0.0)
        return np.array(np.mean(amplitudes))


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class SCRAverageDuration(_SCRFeature):
    def __init__(self, fs=4.0, params=None):
        super().__init__("SCRAverageDuration", fs=fs, params=params)

    def compute_feature(self, scr):
        _, _, _, durations = self._scr_stats(scr)
        if durations.size == 0:
            return np.array(0.0)
        return np.array(np.mean(durations))


def _detect_resp_extrema(signal, fs, min_breath_period_s=1.5):
    x = _as_float1d(signal)
    min_distance = max(1, int(min_breath_period_s * fs))
    if x.size <= min_distance:
        empty = np.array([], dtype=int)
        return empty, empty
    height = np.std(x) * 0.25
    peaks, _ = find_peaks(x, distance=min_distance, height=height)
    troughs, _ = find_peaks(-x, distance=min_distance, height=height)
    return peaks, troughs


def _resp_breath_intervals(signal, fs):
    peaks, _ = _detect_resp_extrema(signal, fs)
    if peaks.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(peaks) / float(fs)


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class BreathingRate(PhysiologicalRepresentation):
    def __init__(self, fs=500.0, params=None):
        super().__init__("BreathingRate")
        if params is not None:
            fs = params.get("fs", fs)
        self.fs = fs

    def compute_feature(self, signal):
        intervals = _resp_breath_intervals(signal, self.fs)
        if intervals.size == 0:
            return np.array(0.0)
        return np.array(60.0 / np.mean(intervals))


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class BreathIntervalRMSSD(PhysiologicalRepresentation):
    def __init__(self, fs=500.0, params=None):
        super().__init__("BreathIntervalRMSSD")
        if params is not None:
            fs = params.get("fs", fs)
        self.fs = fs

    def compute_feature(self, signal):
        intervals = _resp_breath_intervals(signal, self.fs)
        diffs = _successive_diffs(intervals)
        if diffs.size == 0:
            return np.array(0.0)
        return np.array(np.sqrt(np.mean(diffs**2)))


@register_representation([ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class BreathAmplitude(PhysiologicalRepresentation):
    def __init__(self, fs=500.0, params=None):
        super().__init__("BreathAmplitude")
        if params is not None:
            fs = params.get("fs", fs)
        self.fs = fs

    def compute_feature(self, signal):
        x = _as_float1d(signal)
        peaks, troughs = _detect_resp_extrema(x, self.fs)
        if peaks.size == 0 or troughs.size == 0:
            return np.array(0.0)
        return np.array(np.mean(x[peaks]) - np.mean(x[troughs]))
