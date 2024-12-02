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
import cv2
import numpy as np
from scipy.io.wavfile import write
import random
import os

from systemds.scuro.modality.video_modality import VideoModality
from systemds.scuro.modality.audio_modality import AudioModality
from systemds.scuro.modality.text_modality import TextModality


class TestDataGenerator:
    def __init__(self, modalities, path, balanced=True):
        self.modalities = modalities
        self.path = path
        self.balanced = balanced

        for modality in modalities:
            mod_path = f"{self.path}/{modality.name.lower()}/"
            os.mkdir(mod_path)
            modality.file_path = mod_path
        self.labels = []
        self.label_path = f"{path}/labels.npy"

    def create_multimodal_data(self, num_instances, duration=2, seed=42):
        speed_fast = 0
        speed_slow = 0
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
                if isinstance(modality, VideoModality):
                    self.__create_video_data(idx, duration, 30, speed_factor)
                if isinstance(modality, AudioModality):
                    self.__create_audio_data(idx, duration, speed_factor)
                if isinstance(modality, TextModality):
                    self.__create_text_data(idx, speed_factor)

        np.save(f"{self.path}/labels.npy", np.array(self.labels))

    def __create_video_data(self, idx, duration, fps, speed_factor):
        path = f"{self.path}/video/{idx}.mp4"

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
        path = f"{self.path}/text/{idx}.txt"

        with open(path, "w") as f:
            f.write(f"The ball moves at speed factor {speed_factor:.2f}.")

    def __create_audio_data(self, idx, duration, speed_factor):
        path = f"{self.path}/audio/{idx}.wav"
        sample_rate = 44100

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        frequency_variation = random.uniform(200.0, 500.0)
        frequency = 440.0 + frequency_variation * np.sin(
            speed_factor * 2 * np.pi * np.linspace(0, 1, len(t))
        )
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

        write(path, sample_rate, audio_data)
