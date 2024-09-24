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
import collections
import json
from datetime import datetime

from systemds.scuro.representations.average import Average
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.modality.aligned_modality import AlignedModality
from systemds.scuro.modality.text_modality import TextModality
from systemds.scuro.modality.video_modality import VideoModality
from systemds.scuro.modality.audio_modality import AudioModality
from systemds.scuro.representations.unimodal import Pickle, JSON, HDF5, NPY
from systemds.scuro.models.discrete_model import DiscreteModel
from systemds.scuro.aligner.task import Task
from systemds.scuro.aligner.dr_search import DRSearch


class CustomTask(Task):
    def __init__(self, model, labels, train_indices, val_indices):
        super().__init__('CustomTask', model, labels, train_indices, val_indices)

    def run(self, data):
        X_train, y_train, X_test, y_test = self.get_train_test_split(data)
        self.model.fit(X_train, y_train, X_test, y_test)
        score = self.model.test(X_test, y_test)
        return score


labels = []
train_indices = []
val_indices = []

video_path = ''
audio_path = ''
text_path = ''

# Load modalities (audio, video, text)
video = VideoModality(video_path, HDF5(), train_indices)
audio = AudioModality(audio_path, Pickle(), train_indices)
text = TextModality(text_path, NPY(), train_indices)

video.read_all()
audio.read_all()
text.read_all()

modalities = [text, audio, video]

model = DiscreteModel()
custom_task = CustomTask(model, labels, train_indices, val_indices)
representations = [Concatenation(), Average()]

dr_search = DRSearch(modalities, custom_task, representations)
best_representation, best_score, best_modalities = dr_search.fit_random()
aligned_representation = dr_search.transform(modalities)
