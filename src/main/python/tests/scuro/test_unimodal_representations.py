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
import copy
import numpy as np

from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.covarep_audio_features import (
    Spectral,
    RMSE,
    Pitch,
    ZeroCrossing,
)
from systemds.scuro.representations.color_histogram import ColorHistogram
from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.tfidf import TfIdf
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.mfcc import MFCC
from tests.scuro.data_generator import (
    TestDataLoader,
    ModalityRandomDataGenerator,
)
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.tabular_features import TabularFeatures
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.timeseries_representations import (
    Mean,
    Max,
    Min,
    Sum,
    Kurtosis,
    Skew,
    Std,
    RMS,
    ACF,
    FrequencyMagnitude,
    SpectralCentroid,
    Quantile,
    ZeroCrossingRate,
    BandpowerFFT,
    LastValue,
    TransitionCount,
    ObservationDensity,
)
from systemds.scuro.representations.physiological_representations import (
    SDNN,
    RMSSD,
    pNN,
    RRPerMinute,
    HRVBandPower,
    HRVVLF,
    HRVLF,
    HRVHF,
    HRVLFHF,
    PoincareSD1,
    PoincareSD2,
    SCLSlope,
    SCLDynamicRange,
    SCRPeaksPerMinute,
    SCRAverageAmplitude,
    SCRAverageDuration,
    BreathingRate,
    BreathIntervalRMSSD,
    BreathAmplitude,
)

TIMESERIES_REPRESENTATIONS = [
    Mean,
    Min,
    Max,
    Sum,
    Std,
    Skew,
    Quantile,
    Kurtosis,
    RMS,
    ZeroCrossingRate,
    LastValue,
    TransitionCount,
    ObservationDensity,
    ACF,
    FrequencyMagnitude,
    SpectralCentroid,
    BandpowerFFT,
]


ECG_REPRESENTATIONS = [
    SDNN,
    RMSSD,
    pNN,
    RRPerMinute,
    HRVBandPower,
    HRVVLF,
    HRVLF,
    HRVHF,
    HRVLFHF,
    PoincareSD1,
    PoincareSD2,
]
EDA_REPRESENTATIONS = [
    SCLSlope,
    SCLDynamicRange,
    SCRPeaksPerMinute,
    SCRAverageAmplitude,
    SCRAverageDuration,
]
RESPIRATION_REPRESENTATIONS = [
    BreathingRate,
    BreathIntervalRMSSD,
    BreathAmplitude,
]
PHYSIOLOGICAL_REPRESENTATIONS = (
    ECG_REPRESENTATIONS + EDA_REPRESENTATIONS + RESPIRATION_REPRESENTATIONS
)


class TestUnimodalRepresentations(unittest.TestCase):
    test_file_path = None
    mods = None
    text = None
    audio = None
    video = None
    data_generator = None
    num_instances = 0

    @classmethod
    def setUpClass(cls):
        cls.num_instances = 2
        cls.indices = np.array(range(cls.num_instances))

    def _create_audio_modality(self, signal_length=1000):
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, signal_length
        )

        audio = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )
        audio.extract_raw_data()
        return audio

    def test_audio_representation_transform_output_shapes(self):
        audio = self._create_audio_modality(signal_length=200)
        audio_representations = [
            (MFCC(), (2, 12)),
            (MelSpectrogram(), (2, 128)),
            (Spectrogram(), (2, 1025)),
            (Spectral(), (2, 4)),
            (ZeroCrossing(), (2, None)),
            (RMSE(), (2, None)),
            (Pitch(), (2, None)),
        ]

        for representation, expected_shape_signature in audio_representations:
            with self.subTest(representation=representation.name):
                transformed_modality = representation.transform(audio)
                self.assertIsNotNone(transformed_modality.data)
                self.assertEqual(len(transformed_modality.data), self.num_instances)

                for transformed_instance in transformed_modality.data:
                    self.assertEqual(
                        transformed_instance.ndim,
                        expected_shape_signature[0],
                    )
                    if expected_shape_signature[1] is not None:
                        self.assertEqual(
                            transformed_instance.shape[1],
                            expected_shape_signature[1],
                        )
                    self.assertGreater(transformed_instance.shape[0], 0)

    def test_audio_representations(self):
        audio_representations = [
            MFCC(),
            MelSpectrogram(),
            Spectrogram(),
            Spectral(),
            ZeroCrossing(),
            RMSE(),
            Pitch(),
        ]
        audio = self._create_audio_modality(signal_length=200)
        original_data = copy.deepcopy(audio.data)

        for representation in audio_representations:
            r = audio.apply_representation(representation)
            assert r.data is not None
            assert len(r.data) == self.num_instances
            for i in range(self.num_instances):
                assert (audio.data[i] == original_data[i]).all()

    def test_timeseries_representations(self):
        ts_representations = [cls() for cls in TIMESERIES_REPRESENTATIONS]
        ts_data, ts_md = ModalityRandomDataGenerator().create_timeseries_data(
            self.num_instances, 100
        )

        ts = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.AUDIO, ts_data, np.float32, ts_md
            )
        )

        ts.extract_raw_data()
        original_data = copy.deepcopy(ts.data)

        for representation in ts_representations:
            r = ts.apply_representation(representation)
            assert r.data is not None
            assert len(r.data) == self.num_instances
            for i in range(self.num_instances):
                assert (ts.data[i] == original_data[i]).all()

    def _create_timeseries_modality(self, data, metadata):
        modality = UnimodalModality(
            TestDataLoader(
                np.array(range(len(data))),
                None,
                ModalityType.TIMESERIES,
                data,
                np.float32,
                metadata,
            )
        )
        modality.extract_raw_data()
        return modality

    def test_timeseries_output_stats_match_transformed_data(self):
        sequence_length = 64
        ts_data, ts_md = ModalityRandomDataGenerator().create_timeseries_data(
            self.num_instances, sequence_length
        )
        ts = self._create_timeseries_modality(ts_data, ts_md)
        input_stats = RepresentationStats(self.num_instances, (sequence_length,))

        for representation_class in TIMESERIES_REPRESENTATIONS:
            with self.subTest(representation=representation_class.__name__):
                representation = representation_class()
                transformed = ts.apply_representation(representation)
                stats = representation.get_output_stats(input_stats)

                self.assertEqual(stats.num_instances, self.num_instances)
                self.assertEqual(
                    np.array(transformed.data).shape[0], self.num_instances
                )
                self.assertEqual(
                    np.array(transformed.data).shape[1:], tuple(stats.output_shape)
                )

    def test_timeseries_batched_path_matches_per_instance_path(self):
        sequence_length = 48
        ts_data, _ = ModalityRandomDataGenerator().create_timeseries_data(
            self.num_instances, sequence_length
        )
        batch = np.stack(ts_data)

        for representation_class in TIMESERIES_REPRESENTATIONS:
            with self.subTest(representation=representation_class.__name__):
                representation = representation_class()
                batched = np.asarray(representation.compute_features_batched(batch))
                if batched.ndim == 1:
                    batched = batched[:, None]
                per_instance = np.stack(
                    [
                        np.atleast_1d(representation.compute_feature(instance))
                        for instance in ts_data
                    ]
                )
                np.testing.assert_allclose(batched, per_instance, rtol=1e-5, atol=1e-5)

    def test_timeseries_representations_on_variable_length_instances(self):
        lengths = [40 + 7 * i for i in range(self.num_instances)]
        ts_data = [np.random.rand(length).astype(np.float32) for length in lengths]
        ts_md = [
            ModalityType.TIMESERIES.create_metadata(["signal"], instance)
            for instance in ts_data
        ]
        ts = self._create_timeseries_modality(ts_data, ts_md)

        for representation_class in TIMESERIES_REPRESENTATIONS:
            with self.subTest(representation=representation_class.__name__):
                transformed = ts.apply_representation(representation_class())
                self.assertEqual(
                    np.array(transformed.data).shape[0], self.num_instances
                )
                self.assertTrue(np.isfinite(np.array(transformed.data)).all())

        spectrum = ts.apply_representation(FrequencyMagnitude())
        self.assertEqual(np.array(spectrum.data).shape[1], max(lengths) // 2 + 1)

    def test_timeseries_representations_on_degenerate_signals(self):
        for name, instance in [
            ("constant", np.full(32, 3.5, dtype=np.float32)),
            ("zeros", np.zeros(32, dtype=np.float32)),
        ]:
            ts_data = [instance.copy() for _ in range(self.num_instances)]
            ts_md = [
                ModalityType.TIMESERIES.create_metadata(["signal"], d) for d in ts_data
            ]
            ts = self._create_timeseries_modality(ts_data, ts_md)
            for representation_class in TIMESERIES_REPRESENTATIONS:
                if representation_class in (Skew, Kurtosis):
                    continue
                with self.subTest(
                    signal=name, representation=representation_class.__name__
                ):
                    transformed = ts.apply_representation(representation_class())
                    self.assertTrue(np.isfinite(transformed.data).all())

    def test_timeseries_minimum_input_length_preconditions(self):
        for representation_class in TIMESERIES_REPRESENTATIONS:
            representation = representation_class()
            minimum = representation.min_input_length
            with self.subTest(representation=representation_class.__name__):
                self.assertIsNone(
                    representation.check_preconditions(
                        RepresentationStats(self.num_instances, (minimum,))
                    )
                )
                if minimum > 1:
                    rejection = representation.check_preconditions(
                        RepresentationStats(self.num_instances, (minimum - 1,))
                    )
                    self.assertIsNotNone(rejection)
                    self.assertIn(str(minimum), rejection)

    def test_acf_rejects_and_narrows_lags_the_input_cannot_express(self):
        acf = ACF(k=10)
        self.assertIsNotNone(
            acf.check_preconditions(RepresentationStats(self.num_instances, (5,)))
        )
        self.assertIsNone(
            acf.check_preconditions(RepresentationStats(self.num_instances, (50,)))
        )

        candidates = [1, 2, 5, 10, 20]
        self.assertEqual(
            acf.filter_parameter_domain(
                "k", candidates, RepresentationStats(self.num_instances, (5,))
            ),
            [1, 2],
        )
        # Never empty: a node with no candidates left would stop being tunable.
        self.assertEqual(
            acf.filter_parameter_domain(
                "k", [5, 10], RepresentationStats(self.num_instances, (2,))
            ),
            [1],
        )
        # Unrelated parameters pass through untouched.
        self.assertEqual(
            acf.filter_parameter_domain(
                "unrelated", candidates, RepresentationStats(self.num_instances, (5,))
            ),
            candidates,
        )

    def test_spectral_operators_bind_sampling_rate_to_the_input(self):
        for representation in [SpectralCentroid(), BandpowerFFT()]:
            with self.subTest(representation=representation.name):
                representation.configure_for_input(
                    RepresentationStats(self.num_instances, (10,), sampling_rate=250.0)
                )
                self.assertEqual(representation.fs, 250.0)

                # An input that does not know its rate must not reset the bound one.
                representation.configure_for_input(
                    RepresentationStats(self.num_instances, (10,), sampling_rate=None)
                )
                self.assertEqual(representation.fs, 250.0)
                self.assertEqual(representation.get_current_parameters()["fs"], 250.0)

    def test_bandpower_band_is_clamped_to_nyquist(self):
        self.assertEqual(BandpowerFFT(band_low=0.5, band_width=1.0).band_high, 1.0)
        self.assertEqual(BandpowerFFT(band_low=0.0, band_width=0.25).band_high, 0.25)

    def test_quantile_returns_one_column_per_requested_quantile(self):
        quantiles = [0.25, 0.5, 0.75]
        sequence_length = 64
        ts_data, ts_md = ModalityRandomDataGenerator().create_timeseries_data(
            self.num_instances, sequence_length
        )
        ts = self._create_timeseries_modality(ts_data, ts_md)

        quantile = Quantile(quantile=quantiles)
        transformed = ts.apply_representation(quantile)
        stats = quantile.get_output_stats(
            RepresentationStats(self.num_instances, (sequence_length,))
        )
        self.assertEqual(tuple(stats.output_shape), (len(quantiles),))
        self.assertEqual(
            np.array(transformed.data).shape, (self.num_instances, len(quantiles))
        )
        # np.quantile prepends its own axis; the columns must still come back in
        # the requested order rather than transposed.
        np.testing.assert_allclose(
            transformed.data[0],
            np.quantile(ts_data[0], quantiles),
            rtol=1e-5,
            atol=1e-5,
        )

    def _create_physiological_modality(self, data, metadata):
        modality = UnimodalModality(
            TestDataLoader(
                np.array(range(len(data))),
                None,
                ModalityType.PHYSIOLOGICAL,
                data,
                np.float32,
                metadata,
            )
        )
        modality.extract_raw_data()
        return modality

    def test_physiological_representations_output_shapes(self):
        data, md = ModalityRandomDataGenerator().create_physiological_data(
            self.num_instances, 2000, kind="ecg", fs=500.0
        )
        physiological = self._create_physiological_modality(data, md)

        for representation_class in PHYSIOLOGICAL_REPRESENTATIONS:
            with self.subTest(representation=representation_class.__name__):
                transformed = physiological.apply_representation(representation_class())
                transformed_data = np.asarray(transformed.data)
                self.assertEqual(transformed_data.shape, (self.num_instances, 1))
                self.assertTrue(np.isfinite(transformed_data).all())

    def test_ecg_features_recover_the_generated_heart_rate(self):
        """The generator lays down R peaks at 0.7-0.9 s intervals, i.e. 66-86
        bpm. Anything outside that means the detector is locking onto the noise
        floor rather than the beats."""
        fs = 500.0
        data, md = ModalityRandomDataGenerator().create_physiological_data(
            self.num_instances, int(fs * 20), kind="ecg", fs=fs
        )
        physiological = self._create_physiological_modality(data, md)
        transformed_data = np.asarray(physiological.data)
        heart_rate = physiological.apply_representation(RRPerMinute(fs=fs))
        self.assertTrue(
            (
                (np.array(heart_rate.data) > 60.0) & (np.array(heart_rate.data) < 95.0)
            ).all()
        )

        # Jittered intervals mean real, non-zero variability.
        for representation_class in [SDNN, RMSSD, PoincareSD1, PoincareSD2]:
            with self.subTest(representation=representation_class.__name__):
                transformed = physiological.apply_representation(
                    representation_class(fs=fs)
                )
                self.assertTrue((np.array(transformed.data) > 0.0).all())

    def test_eda_features_recover_the_generated_scr_peaks(self):
        """One SCR bump every 10 s is 6 per minute."""
        fs = 4.0
        data, md = ModalityRandomDataGenerator().create_physiological_data(
            self.num_instances, 400, kind="eda", fs=fs
        )
        physiological = self._create_physiological_modality(data, md)

        peaks_per_minute = physiological.apply_representation(SCRPeaksPerMinute(fs=fs))
        np.testing.assert_allclose(peaks_per_minute.data, 6.0)

        for representation_class in [SCRAverageAmplitude, SCRAverageDuration]:
            with self.subTest(representation=representation_class.__name__):
                transformed = physiological.apply_representation(
                    representation_class(fs=fs)
                )
                self.assertTrue((np.array(transformed.data) > 0.0).all())

        # A rising tonic level has a positive slope and a non-degenerate range.
        self.assertTrue(
            (np.array(physiological.apply_representation(SCLSlope()).data) > 0.0).all()
        )
        self.assertTrue(
            (
                np.array(physiological.apply_representation(SCLDynamicRange()).data)
                > 0.0
            ).all()
        )

    def test_respiration_features_recover_the_generated_breathing_rate(self):
        """0.25 Hz is 15 breaths per minute."""
        fs = 500.0
        data, md = ModalityRandomDataGenerator().create_physiological_data(
            self.num_instances, int(fs * 20), kind="resp", fs=fs
        )
        physiological = self._create_physiological_modality(data, md)

        breathing_rate = physiological.apply_representation(BreathingRate(fs=fs))
        np.testing.assert_allclose(breathing_rate.data, 15.0, rtol=0.1)

        amplitude = physiological.apply_representation(BreathAmplitude(fs=fs))
        np.testing.assert_allclose(amplitude.data, 2.0, rtol=0.2)

    def test_scl_slope_recovers_a_known_linear_trend(self):
        ramp = (3.0 * np.arange(100) + 2.0).astype(np.float32)
        np.testing.assert_allclose(SCLSlope().compute_feature(ramp), 3.0, rtol=1e-4)
        np.testing.assert_allclose(
            SCLDynamicRange().compute_feature(np.array([1.0, 5.0, -2.0])), 7.0
        )

    def test_physiological_representations_on_degenerate_signals(self):
        """A flat signal has no beats, no SCRs and no breaths. Every detector
        has to fall back to 0.0 instead of dividing by an empty interval list."""
        for name, instance in [
            ("constant", np.ones(400, dtype=np.float32)),
            ("zeros", np.zeros(400, dtype=np.float32)),
            ("single_sample", np.array([0.5], dtype=np.float32)),
        ]:
            data = [instance.copy() for _ in range(self.num_instances)]
            md = [
                ModalityType.PHYSIOLOGICAL.create_metadata(["signal"], d) for d in data
            ]
            physiological = self._create_physiological_modality(data, md)
            for representation_class in PHYSIOLOGICAL_REPRESENTATIONS:
                with self.subTest(
                    signal=name, representation=representation_class.__name__
                ):
                    transformed = physiological.apply_representation(
                        representation_class()
                    )
                    self.assertTrue(np.isfinite(transformed.data).all())
                    np.testing.assert_allclose(transformed.data, 0.0, atol=1e-6)

    def test_tabular_features(self):
        data_generator = ModalityRandomDataGenerator()
        data_generator.modality_type = ModalityType.EMBEDDING
        rows = [[1.0, 2.0, 3.0] for _ in range(self.num_instances)]

        modality = TransformedModality(data_generator, "test_transformation")
        modality.data = rows
        modality.metadata = [
            ModalityType.EMBEDDING.create_metadata(np.asarray(row)) for row in rows
        ]

        tabular_features = TabularFeatures()
        transformed = tabular_features.transform(modality)
        self.assertEqual(transformed.data.shape, (self.num_instances, 3))
        self.assertEqual(transformed.data.dtype, np.float32)
        np.testing.assert_allclose(transformed.data, np.asarray(rows))

    def test_word2vec_representation(self):
        vector_size = 20
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances, 3
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        transformed = text.apply_representation(W2V(vector_size=vector_size))
        transformed_data = np.asarray(transformed.data)
        self.assertEqual(transformed_data.shape, (self.num_instances, vector_size))
        self.assertTrue(np.isfinite(transformed_data).all())

    def test_audio_representations_on_a_signal_shorter_than_one_frame(self):
        """librosa's default frame is 2048 samples. A shorter instance must
        still come back as a single frame rather than an empty array a
        downstream aggregation would then reduce over nothing."""
        audio = self._create_audio_modality(signal_length=8)

        for representation in [
            MFCC(),
            MelSpectrogram(),
            Spectrogram(),
            Spectral(),
            ZeroCrossing(),
            RMSE(),
            Pitch(),
        ]:
            with self.subTest(representation=representation.name):
                transformed = representation.transform(audio)
                self.assertEqual(len(transformed.data), self.num_instances)
                for instance in transformed.data:
                    self.assertEqual(instance.shape[0], 1)
                    self.assertTrue(np.isfinite(instance).all())

    def test_color_histogram_color_spaces_and_normalization(self):
        image_data, image_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 1, height=8, width=8
        )
        image = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.IMAGE, image_data, np.float32, image_md
            )
        )
        image.extract_raw_data()

        for color_space in ["RGB", "HSV", "GRAY"]:
            with self.subTest(color_space=color_space):
                representation = ColorHistogram(
                    color_space=color_space, bins=4, normalize=True
                )
                transformed = np.asarray(representation.transform(image).data)
                self.assertEqual(
                    transformed.shape,
                    (self.num_instances, representation.calculate_hist_dim()),
                )
                np.testing.assert_allclose(transformed.sum(axis=1), 1.0, rtol=1e-5)

        # A single-colour image puts every pixel in one bin -- the degenerate
        # case a downstream model can learn nothing from.
        uniform = [
            np.full((8, 8, 3), 7, dtype=np.uint8) for _ in range(self.num_instances)
        ]
        uniform_md = [
            ModalityType.IMAGE.create_metadata(8, 8, 3)
            for _ in range(self.num_instances)
        ]
        uniform_image = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.IMAGE, uniform, np.float32, uniform_md
            )
        )
        uniform_image.extract_raw_data()
        histogram = np.asarray(
            ColorHistogram(bins=4, normalize=True).transform(uniform_image).data
        )
        np.testing.assert_array_equal((histogram > 0).sum(axis=1), 1)

    def test_image_representations(self):
        image_data, image_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 1, height=8, width=8
        )

        image = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.IMAGE, image_data, np.float32, image_md
            )
        )

        r = image.apply_representation(ColorHistogram())
        assert r.data is not None
        assert len(r.data) == self.num_instances

    # def test_video_representations(self):
    #     video_representations = [
    #         CLIPVisual(layer_name="post_layernorm"),
    #         I3D(),
    #         X3D(),
    #         VGG19(),
    #         ResNet(),
    #         SwinVideoTransformer(),
    #     ]
    #     video_data, video_md = ModalityRandomDataGenerator().create_visual_modality(
    #         self.num_instances, 25
    #     )
    #     video = UnimodalModality(
    #         TestDataLoader(
    #             self.indices, None, ModalityType.VIDEO, video_data, np.float32, video_md
    #         )
    #     )
    #     for representation in video_representations:
    #         r = video.apply_representation(representation)
    #         assert r.data is not None
    #         assert len(r.data) == self.num_instances

    def test_text_representations(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances, 3
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        for representation in [BoW(2, 2), TfIdf()]:
            r = text.apply_representation(representation)
            assert r.data is not None
            assert len(r.data) == self.num_instances

    def test_chunked_video_representations(self):
        video_data, video_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 30
        )
        video = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.VIDEO, video_data, np.float32, video_md
            )
        )
        r = video.apply_representation(ResNet(model_name="ResNet18"))
        assert r.data is not None
        assert len(r.data) == self.num_instances
        assert len(r.metadata) == self.num_instances


# TODO: the representations still untested here are the ones that download a
# pretrained model at construction time -- Bert, RoBERTa, CLIPText/CLIPVisual,
# GloVe, W2V's larger variants, Wav2Vec, VGG19, X3D and SwinVideoTransformer.
# They need either a cached-model fixture or a network-marked test suite, which
# is also why test_video_representations above is commented out.
if __name__ == "__main__":
    unittest.main()
