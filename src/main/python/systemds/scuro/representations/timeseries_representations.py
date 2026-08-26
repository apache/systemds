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
from scipy import stats

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import (
    CONTAINER_ARRAY,
    CONTAINER_LIST,
    RepresentationStats,
)
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import dense_instance_batch
from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_context_representation_operator,
)

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Precision loss occurred in moment calculation",
    category=RuntimeWarning,
)


class TimeSeriesRepresentation(UnimodalRepresentation):

    def __init__(
        self,
        name,
        parameters=None,
        params=None,
        self_contained=False,
        min_input_length=1,
    ):
        if params is None:
            params = {}
        self.min_input_length = min_input_length
        super().__init__(name, ModalityType.EMBEDDING, parameters, self_contained)

    @staticmethod
    def _input_length(input_stats) -> int:
        shape = getattr(input_stats, "output_shape", ()) or ()
        return int(shape[0]) if shape else 0

    def check_preconditions(self, input_stats):
        length = self._input_length(input_stats)
        if length < self.min_input_length:
            return (
                f"{self.name} needs >= {self.min_input_length} samples, "
                f"input provides {length}"
            )
        return None

    def compute_feature(self, signal):
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_features_batched(self, data):
        return np.asarray(self.compute_feature(data, axis=-1))

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        dtype = modality.metadata[0]["data_layout"]["type"]
        batch = dense_instance_batch(modality.data)
        if batch is not None:
            features = self.compute_features_batched(batch)
            if features.ndim == 1:
                features = features[:, None]
            transformed_modality.data = features.astype(dtype)
            return transformed_modality

        result = []

        for signal in modality.data:
            feature = self.compute_feature(signal)
            result.append(feature)

        maxlen = max(r.size for r in result)
        padded_result = [
            np.pad(r, (0, maxlen - r.size), mode="constant", constant_values=0.0)
            for r in result
        ]
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
        input_bytes = (
            int(input_stats.num_instances)
            * self._num_elements(input_stats.output_shape)
            * np.dtype(np.float32).itemsize
        )
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        batch_bytes = (
            input_bytes
            if getattr(input_stats, "container", CONTAINER_ARRAY) == CONTAINER_LIST
            else 0
        )
        cpu_peak = (
            int((input_bytes + batch_bytes + 3 * output_bytes) * 1.15) + 4 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
@register_context_representation_operator(ModalityType.AUDIO)
class Mean(TimeSeriesRepresentation):
    def __init__(self, params=None):
        super().__init__("Mean")

    def compute_feature(self, signal, axis=-1):
        return np.array(np.mean(signal, axis=axis))


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class Min(TimeSeriesRepresentation):
    def __init__(self, params=None):
        super().__init__("Min")

    def compute_feature(self, signal, axis=-1):
        return np.array(np.min(signal, axis=axis))


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class Max(TimeSeriesRepresentation):
    def __init__(self, params=None):
        super().__init__("Max")

    def compute_feature(self, signal, axis=-1):
        return np.array(np.max(signal, axis=axis))


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class Sum(TimeSeriesRepresentation):
    def __init__(self, params=None):
        super().__init__("Sum")

    def compute_feature(self, signal, axis=-1):
        return np.array(np.sum(signal, axis=axis))


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
@register_context_representation_operator(ModalityType.AUDIO)
class Std(TimeSeriesRepresentation):
    def __init__(self, params=None):
        super().__init__("Std", min_input_length=2)

    def compute_feature(self, signal, axis=-1):
        return np.array(np.std(signal, axis=axis))


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
@register_context_representation_operator(ModalityType.AUDIO)
class Skew(TimeSeriesRepresentation):
    def __init__(self, params=None):
        super().__init__("Skew", min_input_length=3)

    def compute_feature(self, signal, axis=-1):
        return np.array(stats.skew(signal, axis=axis))


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class Quantile(TimeSeriesRepresentation):
    def __init__(self, quantile=0.9, params=None):
        super().__init__(
            "Qunatile", {"quantile": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]}
        )
        if params is not None:
            quantile = params.get("quantile", quantile)
        self.quantile = quantile

    def compute_feature(self, signal, axis=-1):
        return np.array(np.quantile(signal, self.quantile, axis=axis))

    def compute_features_batched(self, data):
        features = np.asarray(np.quantile(data, self.quantile, axis=-1))
        if np.ndim(self.quantile) == 0:
            return features

        return np.moveaxis(features, 0, -1)

    def get_output_stats(self, input_stats):
        n_quantiles = np.atleast_1d(self.quantile).size
        return RepresentationStats(
            input_stats.num_instances,
            (n_quantiles,),
            input_stats.output_shape_is_known,
        )


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
@register_context_representation_operator(ModalityType.AUDIO)
class Kurtosis(TimeSeriesRepresentation):
    min_input_length = 4  # the fourth moment is undefined below four samples

    def __init__(self, params=None):
        super().__init__("Kurtosis")

    def compute_feature(self, signal, axis=-1):
        return np.array(stats.kurtosis(signal, fisher=True, bias=True, axis=axis))


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
@register_context_representation_operator(ModalityType.AUDIO)
class RMS(TimeSeriesRepresentation):
    def __init__(self, params=None):
        super().__init__("RMS")

    def compute_feature(self, signal, axis=-1):
        return np.array(np.sqrt(np.mean(np.square(signal), axis=axis)))


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class ZeroCrossingRate(TimeSeriesRepresentation):
    def __init__(self, params=None):
        super().__init__("ZeroCrossingRate", min_input_length=2)

    def compute_feature(self, signal, axis=-1):
        return np.array(np.sum(np.diff(np.signbit(signal), axis=axis) != 0, axis=axis))


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class LastValue(TimeSeriesRepresentation):
    def __init__(self, params=None):
        super().__init__("LastValue")

    def compute_feature(self, signal, axis=-1):
        return np.take(signal, -1, axis=axis)


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class TransitionCount(TimeSeriesRepresentation):
    def __init__(self, threshold=0.0, params=None):
        super().__init__(
            "TransitionCount",
            parameters={"threshold": [0.0, 1.0, 5.0, 10.0]},
            min_input_length=2,
        )
        if params is not None:
            threshold = params.get("threshold", threshold)
        self.threshold = threshold

    def compute_feature(self, signal, axis=-1):
        return np.array(
            np.sum(np.abs(np.diff(signal, axis=axis)) > self.threshold, axis=axis)
        )


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class ObservationDensity(TimeSeriesRepresentation):
    def __init__(self, params=None):
        super().__init__("ObservationDensity", min_input_length=2)

    def compute_feature(self, signal, axis=-1):
        n = signal.shape[axis]
        transitions = np.sum(np.diff(signal, axis=axis) != 0, axis=axis)
        return np.array((transitions + 1) / n)


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class ACF(TimeSeriesRepresentation):
    def __init__(self, k=1, params=None):
        super().__init__(
            "ACF", {"k": [1, 2, 5, 10, 20, 25, 50, 100, 200, 500]}, min_input_length=2
        )
        if params is not None:
            k = params.get("k", k)
        self.k = k

    def filter_parameter_domain(self, name, values, input_stats):
        if name != "k":
            return values
        length = self._input_length(input_stats)
        if length <= 1:
            return values
        usable = [k for k in values if 0 < int(k) < length]
        return usable or [1]

    def check_preconditions(self, input_stats):
        failure = super().check_preconditions(input_stats)
        if failure:
            return failure
        length = self._input_length(input_stats)
        if int(self.k) >= length:
            return (
                f"ACF lag k={int(self.k)} needs > {int(self.k)} samples, "
                f"input provides {length}"
            )
        return None

    def compute_feature(self, signal, axis=-1):
        x = np.asarray(signal, dtype=np.float64)
        x = x - np.mean(x, axis=axis, keepdims=True)
        k = int(self.k)
        n = x.shape[axis]
        if k <= 0 or k >= n:
            out_shape = list(x.shape)
            del out_shape[axis]
            return np.zeros(out_shape) if out_shape else np.array(0.0)
        den = np.sum(x * x, axis=axis)
        xm = np.moveaxis(x, axis, -1)
        corr = np.sum(xm[..., :-k] * xm[..., k:], axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = corr / den
        bad = ~np.isfinite(den) | np.isclose(den, 0.0)
        out = np.where(bad, 0.0, out)
        return np.asarray(out)

    def get_k_values(self, max_length, percent=0.2, num=10, log=False):
        # TODO: Probably would be useful to invoke this function while tuning the hyperparameters depending on the max length of the singal
        max_k = int(max_length * percent)
        if log:
            k_vals = np.unique(np.logspace(0, np.log10(max_k), num=num, dtype=int))
        else:
            k_vals = np.unique(np.linspace(1, max_k, num=num, dtype=int))
        return k_vals.tolist()


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.TIMESERIES)
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
class FrequencyMagnitude(TimeSeriesRepresentation):
    def __init__(self, params=None, self_contained=True):
        super().__init__("FrequencyMagnitude", min_input_length=2)

    def compute_feature(self, signal, axis=-1):
        return np.array(np.abs(np.fft.rfft(signal, axis=axis)))

    def get_output_stats(self, input_stats):
        n = self._num_elements(input_stats.output_shape)
        out_len = n // 2 + 1 if n > 0 else 0
        return RepresentationStats(
            input_stats.num_instances, (out_len,), input_stats.output_shape_is_known
        )


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
@register_context_representation_operator(ModalityType.TIMESERIES)
class SpectralCentroid(TimeSeriesRepresentation):
    def __init__(self, fs=1.0, params=None):
        super().__init__("SpectralCentroid", min_input_length=2)
        if params is not None:
            fs = params.get("fs", fs)
        self.fs = float(fs)

    def get_current_parameters(self):
        current_params = super().get_current_parameters()
        current_params["fs"] = self.fs
        return current_params

    def configure_for_input(self, input_stats):
        sampling_rate = getattr(input_stats, "sampling_rate", None)
        if sampling_rate:
            self.fs = float(sampling_rate)

    def compute_feature(self, signal, axis=-1):
        signal = np.asarray(signal, dtype=np.float64)
        n = signal.shape[axis]
        frequency_magnitude = FrequencyMagnitude().compute_feature(signal, axis=axis)
        frequencies = np.fft.rfftfreq(n, d=1.0 / self.fs)
        ax = axis if axis >= 0 else frequency_magnitude.ndim + axis
        freq_shape = [1] * frequency_magnitude.ndim
        freq_shape[ax] = frequencies.size
        frequencies = frequencies.reshape(freq_shape)
        num = np.sum(frequencies * frequency_magnitude, axis=axis)
        den = np.sum(frequency_magnitude, axis=axis) + 1e-12
        return np.array(num / den)


@register_representation([ModalityType.TIMESERIES, ModalityType.PHYSIOLOGICAL])
@register_context_representation_operator(ModalityType.PHYSIOLOGICAL)
@register_context_representation_operator(ModalityType.TIMESERIES)
class BandpowerFFT(TimeSeriesRepresentation):
    def __init__(self, fs=1.0, band_low=0.0, band_width=0.5, params=None):
        super().__init__(
            "BandpowerFFT",
            parameters={
                "band_low": [0.0, 0.25, 0.5],
                "band_width": [0.25, 0.5, 1.0],
            },
            min_input_length=2,
        )
        if params is not None:
            fs = params.get("fs", fs)
            band_low = params.get("band_low", band_low)
            band_width = params.get("band_width", band_width)
        self.fs = float(fs)
        self.band_low = float(band_low)
        self.band_width = float(band_width)

    @property
    def band_high(self) -> float:
        return min(1.0, self.band_low + self.band_width)

    def get_current_parameters(self):
        current_params = super().get_current_parameters()
        current_params["fs"] = self.fs  # bound to the data, not searched
        return current_params

    def configure_for_input(self, input_stats):
        sampling_rate = getattr(input_stats, "sampling_rate", None)
        if sampling_rate:
            self.fs = float(sampling_rate)

    def compute_feature(self, signal, axis=-1):
        signal = np.asarray(signal, dtype=np.float64)
        n = signal.shape[axis]
        nyquist = self.fs / 2.0
        self.f1, self.f2 = self.band_low * nyquist, self.band_high * nyquist

        frequency_magnitude = FrequencyMagnitude().compute_feature(signal, axis=axis)
        frequencies = np.fft.rfftfreq(n, d=1.0 / self.fs)

        ax = axis if axis >= 0 else frequency_magnitude.ndim + axis
        freq_shape = [1] * frequency_magnitude.ndim
        freq_shape[ax] = frequencies.size
        frequencies = frequencies.reshape(freq_shape)

        in_band = (frequencies >= self.f1) & (frequencies < self.f2)
        return np.array(np.sum((frequency_magnitude**2) * in_band, axis=axis))
