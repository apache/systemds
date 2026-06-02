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

import time
import unittest
import copy
import numpy as np
from systemds.scuro.representations.bert import (
    Bert,
    ALBERT,
    ELECTRA,
    RoBERTa,
    DistillBERT,
)
from systemds.scuro.representations.clip import CLIPVisual, CLIPText
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.covarep_audio_features import (
    Spectral,
    RMSE,
    Pitch,
    ZeroCrossing,
)
from systemds.scuro.representations.glove import GloVe
from systemds.scuro.representations.wav2vec import Wav2Vec
from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.window_aggregation import WindowAggregation
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.tfidf import TfIdf
from systemds.scuro.representations.x3d import X3D
from systemds.scuro.representations.x3d import I3D
from systemds.scuro.representations.color_histogram import ColorHistogram
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.mfcc import MFCC
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.swin_video_transformer import SwinVideoTransformer
from tests.scuro.data_generator import (
    TestDataLoader,
    ModalityRandomDataGenerator,
)
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.timeseries_representations import (
    Mean,
    Max,
    Min,
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
)
from systemds.scuro.representations.vgg import VGG19


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
        cls.num_instances = 100
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
        audio = self._create_audio_modality()
        audio_representations = [
            (MFCC(), (2, 12)),
            (MelSpectrogram(), (2, 128)),
            (Spectrogram(), (2, 1025)),
            (Wav2Vec(), (1, None)),
            (Spectral(), (2, 4)),
            (ZeroCrossing(), (2, None)),
            (RMSE(), (2, None)),
            (Pitch(), (2, None)),
        ]

        for representation, expected_shape_signature in audio_representations:
            with self.subTest(representation=representation.name):
                transformed_modality = representation.transform(audio)
                print(representation.name)
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
            Wav2Vec(),
            Spectral(),
            ZeroCrossing(),
            RMSE(),
            Pitch(),
        ]
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, 1000
        )

        audio = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )

        audio.extract_raw_data()
        original_data = copy.deepcopy(audio.data)

        for representation in audio_representations:
            r = audio.apply_representation(representation)
            assert r.data is not None
            assert len(r.data) == self.num_instances
            for i in range(self.num_instances):
                assert (audio.data[i] == original_data[i]).all()

    def test_timeseries_representations(self):
        ts_representations = [
            Mean(),
            Max(),
            Min(),
            Kurtosis(),
            Skew(),
            Std(),
            RMS(),
            ACF(),
            FrequencyMagnitude(),
            SpectralCentroid(),
            Quantile(),
            ZeroCrossingRate(),
            BandpowerFFT(),
        ]
        ts_data, ts_md = ModalityRandomDataGenerator().create_timeseries_data(
            self.num_instances, 1000
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

    def test_image_representations(self):
        image_representations = [ColorHistogram(), CLIPVisual(), ResNet()]

        image_data, image_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 1
        )

        image = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.IMAGE, image_data, np.float32, image_md
            )
        )

        for representation in image_representations:
            r = image.apply_representation(representation)
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
        test_representations = [
            CLIPText(),
            Bert(),
            BoW(2, 2),
            TfIdf(),
            W2V(),
            GloVe(),
            ALBERT(),
            ELECTRA(),
            RoBERTa(),
            DistillBERT(),
        ]
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances, 100
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        for representation in test_representations:
            r = text.apply_representation(representation)
            assert r.data is not None
            assert len(r.data) == self.num_instances

    def test_chunked_video_representations(self):
        video_representations = [ResNet()]
        video_data, video_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 25
        )
        video = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.VIDEO, video_data, np.float32, video_md
            )
        )
        for representation in video_representations:
            r = video.apply_representation(representation)
            assert r.data is not None
            assert len(r.data) == self.num_instances
            assert len(r.metadata) == self.num_instances


if __name__ == "__main__":
    unittest.main()
