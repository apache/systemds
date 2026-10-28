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
import unittest

import numpy as np

from systemds.scuro.modality.type import DataLayout, ModalityType
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.mfcc import MFCC
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.window_aggregation import (
    DynamicWindow,
    StaticWindow,
    WindowAggregation,
)


class _FakeModality:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        self.metadata = [
            ModalityType.TIMESERIES.create_metadata(["signal"], instance)
            for instance in self.data
        ]

    def get_data_layout(self):
        return DataLayout.SINGLE_LEVEL


class _BatchedMean(UnimodalRepresentation):
    def __init__(self):
        super().__init__("BatchedMean", ModalityType.EMBEDDING)
        self.scalar_calls = 0
        self.batch_calls = 0

    def compute_feature(self, signal):
        self.scalar_calls += 1
        return np.asarray(signal).mean()

    def compute_features_batched(self, data):
        self.batch_calls += 1
        data = np.asarray(data)
        return data.mean(axis=tuple(range(1, data.ndim)))

    def transform(self, modality, aggregation=None):
        raise NotImplementedError

    def get_output_stats(self, input_stats):
        return RepresentationStats(input_stats.num_instances, (1,))


class TestWindowRepresentationBatching(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.data = rng.normal(size=(4, 100)).astype(np.float32)
        self.modality = _FakeModality(self.data)

    def test_static_window_batches_across_instances_and_windows(self):
        representation = _BatchedMean()
        result = StaticWindow(representation, num_windows=20).execute(self.modality)
        expected = self.data.reshape(4, 20, 5).mean(axis=2)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(representation.scalar_calls, 0)
        self.assertGreater(representation.batch_calls, 0)
        self.assertLess(representation.batch_calls, self.data.shape[0] * 20)

    def test_dynamic_window_batches_equal_shape_windows(self):
        representation = _BatchedMean()
        operator = DynamicWindow(representation, num_windows=16)
        result = operator.execute(self.modality)

        expected = []
        for instance in self.data:
            ends = np.cumsum(operator._window_sizes(len(instance)))
            starts = np.concatenate(([0], ends[:-1]))
            expected.append(
                [instance[start:end].mean() for start, end in zip(starts, ends)]
            )

        np.testing.assert_allclose(result, np.asarray(expected), rtol=1e-6, atol=1e-6)
        self.assertEqual(representation.scalar_calls, 0)
        self.assertGreater(representation.batch_calls, 0)
        self.assertLess(representation.batch_calls, self.data.shape[0] * 16)

    def test_window_aggregation_batches_full_windows_and_preserves_tail(self):
        representation = _BatchedMean()
        operator = WindowAggregation(representation, window_size=7)
        result = operator.execute(self.modality)
        expected = np.stack(
            [
                [
                    instance[start : min(start + 7, len(instance))].mean()
                    for start in range(0, len(instance), 7)
                ]
                for instance in self.data
            ]
        )

        self.assertEqual(result.shape[1], math.ceil(self.data.shape[1] / 7))
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(representation.scalar_calls, 0)

    def test_audio_representations_preserve_results_when_batched(self):
        rng = np.random.default_rng(11)
        windows = rng.normal(size=(3, 64)).astype(np.float32)
        representations = (
            MFCC(n_mfcc=4, n_mels=8, hop_length=8, n_fft=32),
            MelSpectrogram(n_mels=8, hop_length=8, n_fft=32),
            Spectrogram(hop_length=8, n_fft=32),
        )
        for representation in representations:
            with self.subTest(representation=representation.name):
                batched = representation.compute_features_batched(windows)
                per_window = np.stack(
                    [representation.compute_feature(window) for window in windows]
                )
                np.testing.assert_allclose(batched, per_window, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
