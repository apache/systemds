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
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.models.discrete_model import DiscreteModel
from systemds.scuro.drsearch.task import Task
from systemds.scuro.drsearch.dr_search import DRSearch

from systemds.scuro.dataloader.audio_loader import AudioLoader
from systemds.scuro.dataloader.text_loader import TextLoader
from systemds.scuro.dataloader.video_loader import VideoLoader


class CustomTask(Task):
    def __init__(self, model, labels, train_indices, val_indices):
        super().__init__("CustomTask", model, labels, train_indices, val_indices)

    def run(self, data):
        X_train, y_train, X_test, y_test = self.get_train_test_split(data)
        self.model.fit(X_train, y_train, X_test, y_test)
        score = self.model.test(X_test, y_test)
        return score


labels = []
train_indices = []
val_indices = []

all_indices = []

video_path = ""
audio_path = ""
text_path = ""


# Define dataloaders
video_data_loader = VideoLoader(video_path, all_indices, chunk_size=10)
text_data_loader = TextLoader(text_path, all_indices)
audio_data_loader = AudioLoader(audio_path, all_indices)

# Load modalities (audio, video, text)
video = UnimodalModality(video_data_loader, "VIDEO")
audio = UnimodalModality(audio_data_loader, "AUDIO")
text = UnimodalModality(text_data_loader, "TEXT")

# Define unimodal representations
r_v = ResNet()
r_a = MelSpectrogram()
r_t = Bert()

# Transform raw unimodal data
video.apply_representation(r_v)
audio.apply_representation(r_a)
text.apply_representation(r_t)

modalities = [text, audio, video]

model = DiscreteModel()
custom_task = CustomTask(model, labels, train_indices, val_indices)
representations = [Concatenation(), Average()]

dr_search = DRSearch(modalities, custom_task, representations)
best_representation, best_score, best_modalities = dr_search.fit_random()
aligned_representation = dr_search.transform(modalities)
