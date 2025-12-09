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
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.drsearch.operator_registry import register_representation


class TimeSeriesRepresentation(UnimodalRepresentation):
    def __init__(self, name, parameters=None):
        if parameters is None:
            parameters = {}
        super().__init__(name, ModalityType.EMBEDDING, parameters, False)

    def compute_feature(self, signal):
        raise NotImplementedError("Subclasses should implement this method.")

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        result = []

        for signal in modality.data:
            feature = self.compute_feature(signal)
            result.append(feature)

        transformed_modality.data = np.vstack(np.array(result)).astype(
            modality.metadata[list(modality.metadata.keys())[0]]["data_layout"]["type"]
        )
        return transformed_modality


@register_representation([ModalityType.TIMESERIES])
class Mean(TimeSeriesRepresentation):
    def __init__(self):
        super().__init__("Mean")

    def compute_feature(self, signal):
        return np.array(np.mean(signal))


@register_representation([ModalityType.TIMESERIES])
class Min(TimeSeriesRepresentation):
    def __init__(self):
        super().__init__("Min")

    def compute_feature(self, signal):
        return np.array(np.min(signal))


@register_representation([ModalityType.TIMESERIES])
class Max(TimeSeriesRepresentation):
    def __init__(self):
        super().__init__("Max")

    def compute_feature(self, signal):
        return np.array(np.max(signal))


@register_representation([ModalityType.TIMESERIES])
class Sum(TimeSeriesRepresentation):
    def __init__(self):
        super().__init__("Sum")

    def compute_feature(self, signal):
        return np.array(np.sum(signal))


@register_representation([ModalityType.TIMESERIES])
class Std(TimeSeriesRepresentation):
    def __init__(self):
        super().__init__("Std")

    def compute_feature(self, signal):
        return np.array(np.std(signal))


@register_representation([ModalityType.TIMESERIES])
class Skew(TimeSeriesRepresentation):
    def __init__(self):
        super().__init__("Skew")

    def compute_feature(self, signal):
        return np.array(stats.skew(signal))


@register_representation([ModalityType.TIMESERIES])
class Quantile(TimeSeriesRepresentation):
    def __init__(self, quantile=0.9):
        super().__init__(
            "Qunatile", {"quantile": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]}
        )
        self.quantile = quantile

    def compute_feature(self, signal):
        return np.array(np.quantile(signal, self.quantile))


@register_representation([ModalityType.TIMESERIES])
class Kurtosis(TimeSeriesRepresentation):
    def __init__(self):
        super().__init__("Kurtosis")

    def compute_feature(self, signal):
        return np.array(stats.kurtosis(signal, fisher=True, bias=False))


@register_representation([ModalityType.TIMESERIES])
class RMS(TimeSeriesRepresentation):
    def __init__(self):
        super().__init__("RMS")

    def compute_feature(self, signal):
        return np.array(np.sqrt(np.mean(np.square(signal))))


@register_representation([ModalityType.TIMESERIES])
class ZeroCrossingRate(TimeSeriesRepresentation):
    def __init__(self):
        super().__init__("ZeroCrossingRate")

    def compute_feature(self, signal):
        return np.array(np.sum(np.diff(np.signbit(signal)) != 0))


@register_representation([ModalityType.TIMESERIES])
class ACF(TimeSeriesRepresentation):
    def __init__(self, k=1):
        super().__init__("ACF", {"k": [1, 2, 5, 10, 20, 25, 50, 100, 200, 500]})
        self.k = k

    def compute_feature(self, signal):
        x = np.asarray(signal) - np.mean(signal)
        k = int(self.k)
        if k <= 0 or k >= len(x):
            return np.array(0.0)
        den = np.dot(x, x)
        if not np.isfinite(den) or np.isclose(den, 0.0):
            return np.array(0.0)
        corr = np.correlate(x[:-k], x[k:])[0]
        return np.array(corr / den)

    def get_k_values(self, max_length, percent=0.2, num=10, log=False):
        # TODO: Probably would be useful to invoke this function while tuning the hyperparameters depending on the max length of the singal
        max_k = int(max_length * percent)
        if log:
            k_vals = np.unique(np.logspace(0, np.log10(max_k), num=num, dtype=int))
        else:
            k_vals = np.unique(np.linspace(1, max_k, num=num, dtype=int))
        return k_vals.tolist()


@register_representation([ModalityType.TIMESERIES])
class FrequencyMagnitude(TimeSeriesRepresentation):
    def __init__(self):
        super().__init__("FrequencyMagnitude")

    def compute_feature(self, signal):
        return np.array(np.abs(np.fft.rfft(signal)))


@register_representation([ModalityType.TIMESERIES])
class SpectralCentroid(TimeSeriesRepresentation):
    def __init__(self, fs=1.0):
        super().__init__("SpectralCentroid", parameters={"fs": [0.5, 1.0, 2.0]})
        self.fs = fs

    def compute_feature(self, signal):
        frequency_magnitude = FrequencyMagnitude().compute_feature(signal)
        freqencies = np.fft.rfftfreq(len(signal), d=1.0 / self.fs)
        num = np.sum(freqencies * frequency_magnitude)
        den = np.sum(frequency_magnitude) + 1e-12
        return np.array(num / den)


@register_representation([ModalityType.TIMESERIES])
class BandpowerFFT(TimeSeriesRepresentation):
    def __init__(self, fs=1.0, f1=0.0, f2=0.5):
        super().__init__(
            "BandpowerFFT",
            parameters={"fs": [0.5, 1.0], "f1": [0.0, 1.0], "f2": [0.5, 1.0]},
        )
        self.fs = fs
        self.f1 = f1
        self.f2 = f2

    def compute_feature(
        self,
        signal,
    ):
        frequency_magnitude = FrequencyMagnitude().compute_feature(signal)
        freqencies = np.fft.rfftfreq(len(signal), d=1.0 / self.fs)
        m = (freqencies >= self.f1) & (freqencies < self.f2)
        return np.array(np.sum(frequency_magnitude[m] ** 2))
