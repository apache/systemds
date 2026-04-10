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

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from systemds.scuro.dataloader.audio_loader import AudioStats
from systemds.scuro.dataloader.image_loader import ImageStats
from systemds.scuro.dataloader.text_loader import TextStats
from systemds.scuro.dataloader.timeseries_loader import TimeseriesStats
from systemds.scuro.dataloader.video_loader import VideoStats
from systemds.scuro.models.model import Model
from systemds.scuro.dataloader.base_loader import BaseLoader

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.task import Task
from systemds.scuro.representations.representation import RepresentationStats

random_state = 42
np.random.seed(random_state)


class TestDataLoader(BaseLoader):
    def __init__(self, indices, chunk_size, modality_type, data, data_type, metadata):
        super().__init__("", indices, data_type, chunk_size, modality_type)

        self.metadata = metadata
        self.test_data = data
        if modality_type == ModalityType.TEXT:
            self.stats = TextStats(
                len(data),
                max(len(d) for d in data),
                sum(len(d) for d in data) / len(data),
                max(len(d.split(" ")) for d in data),
                sum(len(d.split(" ")) for d in data) / len(data),
                (max(len(d) for d in data),),
            )
        elif modality_type == ModalityType.AUDIO:
            self.stats = AudioStats(
                16000,
                max(len(d) for d in data),
                sum(len(d) for d in data) / len(data),
                len(data),
                output_shape_is_known=True,
            )
        elif modality_type == ModalityType.VIDEO:
            self.stats = VideoStats(
                30,
                max(d.shape[0] for d in data),
                max(d.shape[1] for d in data),
                max(d.shape[2] for d in data),
                max(d.shape[3] for d in data),
                len(data),
            )
        elif modality_type == ModalityType.TIMESERIES:
            self.stats = TimeseriesStats(
                max(len(d) for d in data),
                len(data),
                sum(len(d) for d in data) / len(data),
                (max(len(d) for d in data),),
                True,
            )
        elif modality_type == ModalityType.IMAGE:
            self.stats = ImageStats(
                max(d.shape[0] for d in data),
                max(d.shape[1] for d in data),
                max(d.shape[2] for d in data),
                len(data),
                (
                    max(d.shape[0] for d in data),
                    max(d.shape[1] for d in data),
                    max(d.shape[2] for d in data),
                ),
                average_width=sum(d.shape[0] for d in data) / len(data),
                average_height=sum(d.shape[1] for d in data) / len(data),
                average_channels=sum(d.shape[2] for d in data) / len(data),
            )

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
        np.random.seed(4)
        self.modality_id = 0
        self.modality_type = None
        self.metadata = {}
        self.data_type = np.float32
        self.transform_time = 0
        self.stats = None

    def create_stats(self, data):
        if self.modality_type == ModalityType.TEXT:
            self.stats = TextStats(
                len(data),
                max(len(d) for d in data),
                sum(len(d) for d in data) / len(data),
                max(len(d.split(" ")) for d in data),
                sum(len(d.split(" ")) for d in data) / len(data),
                (max(len(d) for d in data),),
            )
        elif self.modality_type == ModalityType.AUDIO:
            self.stats = AudioStats(
                16000,
                max(len(d) for d in data),
                sum(len(d) for d in data) / len(data),
                len(data),
            )
        elif self.modality_type == ModalityType.VIDEO:
            self.stats = VideoStats(
                30,
                max(d.shape[0] for d in data),
                max(d.shape[1] for d in data),
                max(d.shape[2] for d in data),
                max(d.shape[3] for d in data),
                len(data),
            )
        elif self.modality_type == ModalityType.TIMESERIES:
            self.stats = TimeseriesStats(
                len(data),
                max(len(d) for d in data),
                sum(len(d) for d in data) / len(data),
                (max(len(d) for d in data),),
                True,
            )
        elif self.modality_type == ModalityType.IMAGE:
            self.stats = ImageStats(
                max(d.shape[0] for d in data),
                max(d.shape[1] for d in data),
                max(d.shape[2] for d in data),
                len(data),
                (
                    max(d.shape[0] for d in data),
                    max(d.shape[1] for d in data),
                    max(d.shape[2] for d in data),
                ),
            )
        else:
            raise ValueError(f"Unsupported modality type: {self.modality_type}")
        return self.stats

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
                self.metadata[i] = modality_type.create_metadata(
                    num_features / 10, data[i]
                )
            elif modality_type == ModalityType.TEXT:
                self.metadata[i] = modality_type.create_metadata(
                    num_features / 10, data[i]
                )
            elif modality_type == ModalityType.VIDEO:
                self.metadata[i] = modality_type.create_metadata(
                    num_features / 30, 10, 0, 0, 1
                )
            elif modality_type == ModalityType.TIMESERIES:
                self.metadata[i] = modality_type.create_metadata(["test"], data[i])
            else:
                raise NotImplementedError

        tf_modality = TransformedModality(self, "test_transformation")
        tf_modality.data = data
        self.modality_id += 1
        return tf_modality

    def create_audio_data(self, num_instances, max_audio_length):
        self.modality_type = ModalityType.AUDIO
        data = [
            [
                random.random()
                for _ in range(random.randint(max_audio_length * 0.9, max_audio_length))
            ]
            for _ in range(num_instances)
        ]

        for i in range(num_instances):
            data[i] = np.array(data[i]).astype(self.data_type)

        self.metadata = {
            i: self.modality_type.create_metadata(16000, np.array(data[i]))
            for i in range(num_instances)
        }

        return data, self.metadata

    def create_timeseries_data(self, num_instances, sequence_length, num_features=1):
        self.modality_type = ModalityType.TIMESERIES
        data = [
            np.random.rand(sequence_length, num_features).astype(self.data_type)
            for _ in range(num_instances)
        ]
        if num_features == 1:
            data = [d.squeeze(-1) for d in data]
        self.metadata = {
            i: self.modality_type.create_metadata(
                [f"feature_{j}" for j in range(num_features)], data[i]
            )
            for i in range(num_instances)
        }
        return data, self.metadata

    def create_text_data(self, num_instances, num_sentences_per_instance=1):
        self.modality_type = ModalityType.TEXT
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
        punctuation = [".", "?", "!"]

        sentences = []
        for _ in range(num_instances):
            sentence = ""
            for i in range(num_sentences_per_instance):
                include_adverb = np.random.random() < 0.7

                subject = np.random.choice(subjects)
                verb = np.random.choice(verbs)
                obj = np.random.choice(objects)
                adverb = np.random.choice(adverbs) if include_adverb else ""
                punct = np.random.choice(punctuation)

                sentence += " " if i > 0 else ""
                sentence += f"{subject}"
                sentence += f" {adverb}" if include_adverb else ""
                sentence += f" {verb} {obj}{punct}"
            sentences.append(sentence)

        self.metadata = {
            i: self.modality_type.create_metadata(len(sentences[i]), sentences[i])
            for i in range(num_instances)
        }

        return sentences, self.metadata

    def create_3d_modality(self, num_instances, dims=(100, 28, 28)):
        self.modality_type = ModalityType.EMBEDDING
        data = [
            np.random.rand(dims[0], dims[1], dims[2]).astype(self.data_type)
            for _ in range(num_instances)
        ]
        self.metadata = {
            i: self.modality_type.create_metadata(data[i]) for i in range(num_instances)
        }
        return data, self.metadata

    def create_2d_modality(self, num_instances, dims=(100, 28)):
        self.modality_type = ModalityType.EMBEDDING
        data = [
            np.random.rand(dims[0], dims[1]).astype(self.data_type)
            for _ in range(num_instances)
        ]
        self.metadata = {
            i: self.modality_type.create_metadata(data[i]) for i in range(num_instances)
        }
        return data, self.metadata

    def create_visual_modality(
        self, num_instances, max_num_frames=1, height=28, width=28, color_channels=3
    ):
        if max_num_frames > 1:
            self.modality_type = ModalityType.VIDEO
            data = [
                np.random.uniform(
                    0.0,
                    1.0,
                    (
                        np.random.randint(10, max_num_frames + 1),
                        height,
                        width,
                        color_channels,
                    ),
                )
                for _ in range(num_instances)
            ]

            self.metadata = {
                i: self.modality_type.create_metadata(
                    30, data[i].shape[0], width, height, color_channels
                )
                for i in range(num_instances)
            }
        else:
            self.modality_type = ModalityType.IMAGE
            data = [
                np.random.randint(
                    0,
                    256,
                    (height, width, color_channels),
                    dtype=np.uint8,
                )
                for _ in range(num_instances)
            ]
            self.metadata = {
                i: self.modality_type.create_metadata(width, height, color_channels)
                for i in range(num_instances)
            }

        return data, self.metadata

    def create_balanced_labels(self, num_instances, num_classes=2):
        if num_instances % num_classes != 0:
            raise ValueError("Size must be even to have equal numbers of classes.")

        class_size = int(num_instances / num_classes)
        vector = np.array([0] * class_size)
        for i in range(num_classes - 1):
            vector = np.concatenate((vector, np.array([1] * class_size)))

        np.random.shuffle(vector)
        return vector


def setup_data(modality_types, num_instances, path):
    if os.path.isdir(path):
        shutil.rmtree(path)

    os.makedirs(path)

    indizes = [str(i) for i in range(0, num_instances)]

    data_generator = TestDataGenerator(modality_types, path)
    data_generator.create_multimodal_data(num_instances)
    return data_generator


class TestDataGenerator:
    def __init__(self, modalities_types, path, balanced=True):

        self.modalities_types = modalities_types
        self.modalities_paths = {
            modality_type.name: f"{path}/{modality_type.name}/"
            for modality_type in modalities_types
        }
        self._indices = None
        self.path = path
        self.balanced = balanced

        for modality_path in self.modalities_paths.values():
            os.mkdir(modality_path)

        self.labels = []
        self.label_path = f"{path}/labels.npy"

    def get_modality_path(self, modality_type):
        return self.modalities_paths[modality_type.name]

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

            for modality_type in self.modalities_types:
                if modality_type == ModalityType.VIDEO:
                    self.__create_visual_data(
                        idx, duration, 30, speed_factor, modality_type
                    )
                elif modality_type == ModalityType.IMAGE:
                    self.__create_visual_data(idx, 1, 1, speed_factor, modality_type)
                elif modality_type == ModalityType.AUDIO:
                    self.__create_audio_data(idx, duration, speed_factor)
                elif modality_type == ModalityType.TEXT:
                    self.__create_text_data(idx, speed_factor)
                elif modality_type == ModalityType.TIMESERIES:
                    self.__create_timeseries_data(idx, duration, speed_factor)
                else:
                    raise ValueError(f"Unsupported modality type: {modality_type}")

        np.save(self.label_path, np.array(self.labels))

    def __create_timeseries_data(self, idx, duration, speed_factor):
        path = f"{self.path}/TIMESERIES/{idx}.npy"
        data = np.random.rand(duration, 1)
        np.save(path, data)

    def __create_visual_data(self, idx, duration, fps, speed_factor, modality_type):
        if modality_type == ModalityType.VIDEO:
            ext = "mp4"
        elif modality_type == ModalityType.IMAGE:
            ext = "jpg"
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")

        path = f"{self.path}/{modality_type.name}/{idx}.{ext}"

        width, height = 160, 120
        if modality_type == ModalityType.VIDEO:
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
        elif modality_type == ModalityType.IMAGE:
            out = cv2.imwrite(path, np.ones((height, width, 3), dtype=np.uint8) * 255)
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")

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


class TestSVM(Model):
    def __init__(self, name):
        super().__init__(name)

    def fit(self, X, y, X_test, y_test):
        return {"accuracy": random.uniform(0.5, 1.0)}, 0

    def test(self, test_X: np.ndarray, test_y: np.ndarray):
        return {"accuracy": random.uniform(0.5, 1.0)}, 0


class TestTask(Task):
    def __init__(self, name, model_name, num_instances):
        self.labels = ModalityRandomDataGenerator().create_balanced_labels(
            num_instances=num_instances
        )
        split = train_test_split(
            np.array(range(num_instances)),
            self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels,
        )
        self.train_indizes, self.val_indizes = [int(i) for i in split[0]], [
            int(i) for i in split[1]
        ]

        super().__init__(
            name,
            TestSVM(model_name),
            self.labels,
            self.train_indizes,
            self.val_indizes,
        )

    def get_output_stats(self, input_stats):
        return RepresentationStats(1, (1,))
