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
import shutil

import cv2
import numpy as np
from scipy.io.wavfile import write
import random
import os

import nltk

from systemds.scuro.dataloader.base_loader import BaseLoader
from systemds.scuro.dataloader.video_loader import VideoLoader
from systemds.scuro.dataloader.audio_loader import AudioLoader
from systemds.scuro.dataloader.text_loader import TextLoader
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType


class TestDataLoader(BaseLoader):
    def __init__(self, indices, chunk_size, modality_type, data, data_type, metadata):
        super().__init__("", indices, data_type, chunk_size, modality_type)

        self.metadata = metadata
        self.test_data = data

    def extract(self, file, indices):
        self.data = self.test_data


class ModalityRandomDataGenerator:

    def __init__(self):
        self.modality_id = 0
        self.modality_type = None
        self.metadata = {}
        self.data_type = np.float32

    def create1DModality(
        self,
        num_instances,
        num_features,
        modality_type,
    ):
        data = np.random.rand(num_instances, num_features).astype(self.data_type)
        data.dtype = self.data_type

        # TODO: write a dummy method to create the same metadata for all instances to avoid the for loop
        self.modality_type = modality_type
        for i in range(num_instances):
            if modality_type == ModalityType.AUDIO:
                self.metadata[i] = modality_type.create_audio_metadata(
                    num_features / 10, data[i]
                )
            elif modality_type == ModalityType.TEXT:
                self.metadata[i] = modality_type.create_text_metadata(
                    num_features / 10, data[i]
                )
            elif modality_type == ModalityType.VIDEO:
                self.metadata[i] = modality_type.create_video_metadata(
                    num_features / 30, 10, 0, 0, 1
                )
            else:
                raise NotImplementedError

        tf_modality = TransformedModality(self, "test_transformation")
        tf_modality.data = data
        self.modality_id += 1
        return tf_modality

    def create_audio_data(self, num_instances, num_features):
        data = np.random.rand(num_instances, num_features).astype(np.float32)
        metadata = {
            i: ModalityType.AUDIO.create_audio_metadata(num_features / 10, data[i])
            for i in range(num_instances)
        }

        return data, metadata

    def create_text_data(self, num_instances):
        nltk.download("webtext")
        sentences = nltk.corpus.webtext.sents()[:num_instances]

        metadata = {
            i: ModalityType.TEXT.create_text_metadata(len(sentences[i]), sentences[i])
            for i in range(num_instances)
        }

        return [" ".join(sentence) for sentence in sentences], metadata

    def create_visual_modality(self, num_instances, num_frames=1, height=28, width=28):
        if num_frames == 1:
            print(f"TODO: create image metadata")
        else:
            metadata = {
                i: ModalityType.VIDEO.create_video_metadata(
                    num_instances / 30, num_frames / 30, width, height, 1
                )
                for i in range(num_instances)
            }

        return (
            np.random.randint(
                0, 256, (num_instances, num_frames, height, width)
            ).astype(np.float16),
            metadata,
        )


def setup_data(modalities, num_instances, path):
    if os.path.isdir(path):
        shutil.rmtree(path)

    os.makedirs(path)

    indizes = [str(i) for i in range(0, num_instances)]

    modalities_to_create = []
    for modality in modalities:
        mod_path = path + "/" + modality.name + "/"

        if modality == ModalityType.VIDEO:
            data_loader = VideoLoader(mod_path, indizes)
        elif modality == ModalityType.AUDIO:
            data_loader = AudioLoader(mod_path, indizes)
        elif modality == ModalityType.TEXT:
            data_loader = TextLoader(mod_path, indizes)
        else:
            raise "Modality not supported in DataGenerator"

        modalities_to_create.append(UnimodalModality(data_loader))

    data_generator = TestDataGenerator(modalities_to_create, path)
    data_generator.create_multimodal_data(num_instances)
    return data_generator


class TestDataGenerator:
    def __init__(self, modalities, path, balanced=True):

        self.modalities = modalities
        self.modalities_by_type = {}
        for modality in modalities:
            self.modalities_by_type[modality.modality_type] = modality

        self._indices = None
        self.path = path
        self.balanced = balanced

        for modality in modalities:
            mod_path = f"{self.path}/{modality.modality_type.name}/"
            os.mkdir(mod_path)
            modality.file_path = mod_path
        self.labels = []
        self.label_path = f"{path}/labels.npy"

    def get_modality_path(self, modality_type):
        return self.modalities_by_type[modality_type].data_loader.source_path

    @property
    def indices(self):
        if self._indices is None:
            raise "No indices available, please call setup_data first"
        return self._indices

    def create_multimodal_data(self, num_instances, duration=2, seed=42):
        speed_fast = 0
        speed_slow = 0
        self._indices = [str(i) for i in range(0, num_instances)]
        for idx in range(num_instances):
            np.random.seed(seed)
            if self.balanced:
                inst_half = int(num_instances / 2)
                if speed_slow < inst_half and speed_fast < inst_half:
                    speed_factor = random.uniform(0.5, 1.5)
                elif speed_fast >= inst_half:
                    speed_factor = random.uniform(0.5, 0.99)
                else:
                    speed_factor = random.uniform(1, 1.5)

            else:
                if speed_fast >= int(num_instances * 0.9):
                    speed_factor = random.uniform(0.5, 0.99)
                elif speed_slow >= int(num_instances * 0.9):
                    speed_factor = random.uniform(0.5, 1.5)
                else:
                    speed_factor = random.uniform(1, 1.5)

            self.labels.append(1 if speed_factor >= 1 else 0)

            if speed_factor >= 1:
                speed_fast += 1
            else:
                speed_slow += 1

            for modality in self.modalities:
                if modality.modality_type == ModalityType.VIDEO:
                    self.__create_video_data(idx, duration, 30, speed_factor)
                if modality.modality_type == ModalityType.AUDIO:
                    self.__create_audio_data(idx, duration, speed_factor)
                if modality.modality_type == ModalityType.TEXT:
                    self.__create_text_data(idx, speed_factor)

        np.save(f"{self.path}/labels.npy", np.array(self.labels))

    def __create_video_data(self, idx, duration, fps, speed_factor):
        path = f"{self.path}/VIDEO/{idx}.mp4"

        width, height = 160, 120
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))

        num_frames = duration * fps
        ball_radius = 20
        center_x = width // 2

        amplitude = random.uniform(0.5, 1.5) * (height // 3)

        for i in range(num_frames):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            center_y = int(
                height // 2
                + amplitude * np.sin(speed_factor * 2 * np.pi * i / num_frames)
            )
            frame = cv2.circle(
                frame, (center_x, center_y), ball_radius, (0, 255, 0), -1
            )
            out.write(frame)

        out.release()

    def __create_text_data(self, idx, speed_factor):
        path = f"{self.path}/TEXT/{idx}.txt"

        with open(path, "w") as f:
            f.write(f"The ball moves at speed factor {speed_factor:.2f}.")

    def __create_audio_data(self, idx, duration, speed_factor):
        path = f"{self.path}/AUDIO/{idx}.wav"
        sample_rate = 16000

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        frequency_variation = random.uniform(200.0, 500.0)
        frequency = 440.0 + frequency_variation * np.sin(
            speed_factor * 2 * np.pi * np.linspace(0, 1, len(t))
        )
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

        write(path, sample_rate, audio_data)
