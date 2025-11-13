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

    def reset(self):
        self._next_chunk = 0
        self.data = []

    def extract(self, file, indices):
        if isinstance(self.test_data, list):
            self.data = [self.test_data[i] for i in indices]
        else:
            self.data = self.test_data[indices]


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
            elif modality_type == ModalityType.TIMESERIES:
                self.metadata[i] = modality_type.create_ts_metadata(["test"], data[i])
            else:
                raise NotImplementedError

        tf_modality = TransformedModality(self, "test_transformation")
        tf_modality.data = data
        self.modality_id += 1
        return tf_modality

    def create_audio_data(self, num_instances, max_audio_length):
        data = [
            [
                random.random()
                for _ in range(random.randint(max_audio_length * 0.9, max_audio_length))
            ]
            for _ in range(num_instances)
        ]

        for i in range(num_instances):
            data[i] = np.array(data[i]).astype(self.data_type)

        metadata = {
            i: ModalityType.AUDIO.create_audio_metadata(16000, np.array(data[i]))
            for i in range(num_instances)
        }

        return data, metadata

    def create_timeseries_data(self, num_instances, sequence_length, num_features=1):
        data = [
            np.random.rand(sequence_length, num_features).astype(self.data_type)
            for _ in range(num_instances)
        ]
        if num_features == 1:
            data = [d.squeeze(-1) for d in data]
        metadata = {
            i: ModalityType.TIMESERIES.create_ts_metadata(
                [f"feature_{j}" for j in range(num_features)], data[i]
            )
            for i in range(num_instances)
        }
        return data, metadata

    def create_text_data(self, num_instances):
        subjects = [
            "The cat",
            "A dog",
            "The student",
            "The teacher",
            "The bird",
            "The child",
            "The programmer",
            "The scientist",
            "A researcher",
        ]
        verbs = [
            "reads",
            "writes",
            "studies",
            "analyzes",
            "creates",
            "develops",
            "designs",
            "implements",
            "examines",
        ]
        objects = [
            "the document",
            "the code",
            "the data",
            "the problem",
            "the solution",
            "the project",
            "the research",
            "the paper",
        ]
        adverbs = [
            "carefully",
            "quickly",
            "efficiently",
            "thoroughly",
            "diligently",
            "precisely",
            "methodically",
        ]

        sentences = []
        for _ in range(num_instances):
            include_adverb = np.random.random() < 0.7

            subject = np.random.choice(subjects)
            verb = np.random.choice(verbs)
            obj = np.random.choice(objects)
            adverb = np.random.choice(adverbs) if include_adverb else ""

            sentence = f"{subject} {adverb} {verb} {obj}"

            sentences.append(sentence)

        metadata = {
            i: ModalityType.TEXT.create_text_metadata(len(sentences[i]), sentences[i])
            for i in range(num_instances)
        }

        return sentences, metadata

    def create_visual_modality(
        self, num_instances, max_num_frames=1, height=28, width=28
    ):
        data = [
            np.random.randint(
                0,
                256,
                (np.random.randint(5, max_num_frames + 1), height, width, 3),
                dtype=np.uint8,
            )
            for _ in range(num_instances)
        ]
        if max_num_frames == 1:
            print(f"TODO: create image metadata")
        else:
            metadata = {
                i: ModalityType.VIDEO.create_video_metadata(
                    30, data[i].shape[0], width, height, 3
                )
                for i in range(num_instances)
            }

        return (data, metadata)

    def create_balanced_labels(self, num_instances, num_classes=2):
        if num_instances % num_classes != 0:
            raise ValueError("Size must be even to have equal numbers of classes.")

        class_size = int(num_instances / num_classes)
        vector = np.array([0] * class_size)
        for i in range(num_classes - 1):
            vector = np.concatenate((vector, np.array([1] * class_size)))

        np.random.shuffle(vector)
        return vector


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
